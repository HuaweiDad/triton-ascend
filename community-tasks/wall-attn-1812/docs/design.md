# Wall Attention 算子 Ascend 适配设计文档

> 社区任务：[triton-lang/triton-ascend#1812](https://github.com/triton-lang/triton-ascend/issues/1812)【社区任务】Wall Attention 算子 Ascend 适配
> 上游参考：[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/ops/wall_attn`（Tilde Research 贡献：Timor Averbuch, Dhruv Pai）
> 目标环境：Triton-Ascend == 3.2.2，torch-npu == 2.9.0，CANN 9.1.0；目标平台：Ascend A2 / A3 / 950

---

## 一、需求背景

### 1.1 需求来源

本任务来自昇腾社区任务池（Issue #1812），要求基于 triton-ascend 在昇腾 NPU 上适配上游 flash-linear-attention（下称 fla）中的 Wall Attention 算子，支持训练并行前向/反向及推理 decode 路径，最终向 GitCode `Ascend/triton-ascend-kernels`（experimental 分支）提交 PR 并合入。

验收标准（摘自任务 Issue）：

1. Triton-Ascend 3.2.2 + torch_npu 环境下 `pytest tests/ops/test_wall_attn.py` 全量通过（0 failed、0 error），**不得修改断言与容差**；
2. 覆盖 Ascend A2、A3、950 三个平台；
3. 消除 aicore / vector core 超时（错误码 507014 / 507034）及编译器崩溃；
4. 前向与反向梯度数值精度满足 `assert_close` 容差；
5. 固定长度与 varlen、fp16 与 fp32 全部参数化配置通过；
6. 不引入回归；PR 需附根因分析、修改说明与验证结果。

### 1.2 背景介绍

#### 1.2.1 Wall Attention 算子简介

Wall Attention 是 Tilde Research 提出并贡献到 fla 的一种**带逐通道门控衰减的因果注意力**，可理解为 "Gated DeltaNet 系" 思想在全注意力上的推广：在标准 QK^T 打分中，对**每个特征通道 n** 引入随时间距离指数衰减的乘性门控，使模型获得数据相关的、细粒度的遗忘能力。同时支持：

- **sink bias**：注意力汇（attention sink）偏置，在 softmax 分母中注入每头一个的虚拟 logit，稳定长序列 softmax 分布（StreamingLLM 类技术）；
- **scalar gate**（FoX 式加性标量门）：在 logit 上叠加 `c_i − c_j` 形式的标量门控差；
- **滑动窗口**（sliding window）因果掩码；
- **varlen**：基于 `cu_seqlens` 的变长序列打包（batch 内多序列不串扰）；
- **GQA**：查询头数 HQ 为 KV 头数 H 的整数倍；
- 三条执行路径：训练 **parallel**（前向 + 反向）、推理 **decode**（预缩放 KV cache 单步解码）、以及纯 PyTorch 的 **naive** 精确参考实现（精度对拍 oracle）。

#### 1.2.2 输入输出语义

**训练 / prefill 路径 `parallel_wall_attn`**：

| 参数 | 形状 | 类型 | 含义 | 约束 |
|------|------|------|------|------|
| q | [B, T, HQ, K] | fp16/bf16/fp32 | 查询 | 末维连续 |
| k | [B, T, H, K] | 同 q | 键 | HQ % H == 0（GQA） |
| v | [B, T, H, V] | 同 q | 值 | — |
| g | [B, T, HQ, K] | fp32 | 逐通道 log 衰减门（g ≤ 0，由 logsigmoid 产生） | kernel 内部做前缀和 |
| g_scalar | [B, T, HQ] 或 None | fp32 | FoX 式标量门 | 可选 |
| sink_bias | [HQ] 或 None | fp32 | 注意力汇 logit（自然对数域） | 可选 |
| scale | float | — | softmax 缩放，缺省 K**-0.5 | — |
| window_size | int 或 None | — | 滑动窗口宽 W（保留 i−j < W） | 可选 |
| cu_seqlens | [N+1] 或 None | int64/int32 | varlen 累积序列长度 | 要求 B == 1 |
| 输出 o | [B, T, HQ, V] | 同 v | 注意力输出 | — |

反向输出梯度：`dq [B,T,HQ,K]`、`dk [B,T,H,K]`、`dv [B,T,H,V]`、`dg [B,T,HQ,K]`、可选 `dsink_bias [HQ]`、可选 `dg_scalar [B,T,HQ]`。

**推理 decode 路径**（与训练接口不同构，由 modeling 侧选择调用）：

- `build_wall_kv_cache(k, g_cumsum, chunk_size)` → `(k_tilde [B,T,HQ,K], r_cache [B,NC,HQ,K])`：预缩放 KV cache 构建（纯 PyTorch，host 侧）；
- `parallel_wall_attn_decode(q, v, p_curr, k_tilde, r_cache, sink_bias, scale, cache_chunk_size, g_scalar_cumsum)` → `(o [B,T_q,HQ,V], lse)`：单步/短查询解码，查询默认位于 cache 之后，cache 内无因果掩码。

#### 1.2.3 数学原理

令 `RCP_LN2 = 1/ln2`。对 g 做 log2 域前缀和 `P = cumsum(g) · RCP_LN2`（逐通道），对 g_scalar 同理得 `c`。对同一序列内 query 位置 i、key 位置 j（j ≤ i，启用窗口时另要求 i−j < W）：

```
s_ij = scale · RCP_LN2 · Σ_n q_in · k_jn · 2^(P_in − P_jn)     [+ (c_i − c_j)]
```

归一化采用 base-2 online softmax：

```
m_i   = max_j s_ij
p_ij  = 2^(s_ij − m_i)
den_i = Σ_j p_ij  [+ 2^(sink_bias_h·RCP_LN2 − m_i)]            （sink 项可选）
o_i   = Σ_j (p_ij / den_i) · v_j
lse_i = m_i + log2(den_i)
```

**数值稳定性的核心技巧（log 域参考点 R 分解）**：由于 P 单调不增（g ≤ 0），对每个 query 块取块首参考点 `R_n = P_{i_t·BT, n}`，将 `2^(P_in − P_jn)` 分解为 `2^(P_in − R_n) · 2^(R_n − P_jn)`，分别吸收进 q、k（记为 q_til、k_til），使两个指数因子均 ≤ 1，从而 **q_til/k_til 可安全降至 bf16/fp16 走 Cube 张量核**，logit 经 `scale·RCP_LN2` 还原，数学上与精确式等价。对角块（含跨块参考点失效区域）改用更小的子块局部参考点，并对指数做 110 上限 clamp 防 fp32 溢出。

**反向**：在 flash-attention 标准反向（preprocess 计算 δ = rowsum(o∘do)，dq/dkv 两 kernel）基础上扩展门控链路：

- 核内结构性导出 `dP_q = ln2 · q ⊙ dq`、`dP_k = −ln2 · k ⊙ dk`（省一路 [BT,BK] fp32 累加器）；
- host 侧对 dP 做 **reverse cumsum** 并乘 RCP_LN2 得 dg；dg_scalar 同理；
- `dsink_bias = −Σ_{b,t} 2^(sink − lse) · δ`。

**decode**：cache 按每 C 个 token 一个锚点 `R_c = P[chunk_start]` 预缩放：`k_tilde[j] = k[j]·2^(R_c(j) − P_j)`，键的衰减被冻结在存储时刻；查询侧用当前行 `p_curr` 与锚点差值即时 rescale，单 kernel 完成 online softmax 累积。长上下文下指数有界（测试覆盖 T=4096 bf16）。

#### 1.2.4 上游参考实现分析

上游代码位于 `fla/ops/wall_attn/`（parallel.py 1101 行、decode.py 296 行、naive.py 81 行），共 **6 个 Triton kernel + 3 个 host 接口**：

| 组件 | 文件 | 说明 |
|------|------|------|
| `parallel_wall_attn_fwd_kernel` | parallel.py | 训练前向。3D grid (NV, NT, B·HQ)；off-diag 循环（块参考点 R，bf16 tensor core）+ diag 循环（子块局部参考点 + clamp）；`USE_SINK_BIAS/USE_WINDOW/IS_VARLEN/USE_SCALAR_G` 四个 constexpr 分支由 heuristics 推导；autotune 扫 BT∈{64,128}×BS∈{32,64}×warps×stages，按 `BT·BV ≤ 16384` 剪枝 |
| `parallel_wall_attn_bwd_kernel_preprocess` | parallel.py | 反向预处理 δ = rowsum(o∘do)，1D grid |
| `parallel_wall_attn_bwd_kernel_dq` | parallel.py | dq + dg（query 侧），同时输出 dg_scalar 偏量 |
| `parallel_wall_attn_bwd_kernel_dkv` | parallel.py | dk/dv + dg（key 侧），对角块提供 `DIAG_BF16` 快/精双路径（环境变量 `WALL_ATTN_DKV_DIAG_BF16` 控制，默认 bf16） |
| `parallel_wall_attn_decode_kernel` | decode.py | decode 单 kernel，1D 展平 grid，按 cache chunk 循环 |
| `build_wall_kv_cache` | decode.py | 纯 PyTorch cache 构建（repeat_interleave + exp2 广播） |
| `WallParallelAttentionFunction` | parallel.py | autograd.Function：host 侧先做 g/g_scalar 的 `chunk_global_cumsum`，sink_bias 预乘 RCP_LN2；反向末段做 reverse cumsum 得 dg/dg_scalar |
| `naive_wall_attn` | naive.py | 精确 eager 参考（fp32 打分 + 显式掩码），测试对拍 oracle |

关键工程细节：

- **varlen**：host 侧 `prepare_chunk_indices(cu_seqlens, BT=128)` 预生成 (序列号, 块号) 映射，kernel 内据此取 bos/eos；为避免 autotune 扫描 BT 使映射失效，**varlen 固定 BT=128、绕过 autotuner 直接以启发式配置启动**（BS 随 T 收敛到 16–64）；
- **V 维切分（NV > 1）**：V 大于 BV 时按 value 维切多 program，dq/dk/dv/dg 以 fp32 部分和写出再在 host 侧 `sum(0)` 归约；LSE 仅由 `i_v == 0` 的 program 写出；
- **GQA**：kernel 内 `i_h = i_hq // G` 直接映射 KV 头，反向 dk/dv 在 host 侧按 G 分组求和（上游用 einops.reduce，适配时改为 torch 原生以去依赖）；
- **依赖的 fla 公共件**：`chunk_global_cumsum`（本身是 2 个 Triton kernel：scalar/vector 全局扫描）、`prepare_chunk_indices`、`fla.ops.utils.op.exp2/log2`、`fla.utils`（contiguous、autocast_custom_fwd/bwd、autotune_cache_kwargs、check_shared_mem、assert_close、device、input_guard）、`fla.ops.backends.dispatch`（GPU 后端分发，NPU 适配时移除）。

#### 1.2.5 训练前向 kernel 流程图

```mermaid
flowchart TD
    A([program 开始<br/>i_v,i_t,i_bh = program_id]) --> B[varlen? 读 chunk_indices<br/>得 i_n, bos, eos, T]
    B --> C[加载 q, P_q=g_cumsum 块,<br/>块首参考点 R]
    C --> D["q_til = q·2^(P_q − R)<br/>降至输入 dtype（指数 ≤ 0 安全）"]
    D --> E[初始化 online softmax:<br/>m=−inf, acc=0, o=0]
    E --> F["off-diag 循环 i_s ∈ [i_start, i_t·BT)<br/>k_til = k·2^(R − P_k)"]
    F --> G["s = dot(q_til, k_til)·scale·RCP_LN2<br/>+ scalar gate / window 掩码"]
    G --> H[online softmax 更新 m, acc<br/>o += dot(p, v)]
    H --> I["diag 循环 i_s ∈ [i_t·BT, 块尾)<br/>子块局部参考点 + clamp 110<br/>+ 因果掩码"]
    I --> J[同 G/H 累积]
    J --> K{sink_bias?}
    K -->|是| L["acc += 2^(sink − m)"]
    K -->|否| M
    L --> M["o /= acc; lse = m + log2(acc)"]
    M --> N[([写出 o；i_v==0 时写 lse])]
```

## 二、需求分析

### 2.1 外部组件依赖

| 依赖 | 版本 | 用途 | 备注 |
|------|------|------|------|
| triton-ascend | 3.2.2 | Triton kernel 编译执行（Ascend 后端） | 验收指定版本 |
| torch + torch_npu | 2.9.0 | 张量与 NPU 运行时 | 验收指定版本 |
| CANN | 9.1.0 | NPU 驱动/运行时 | 验收指定版本 |
| fla（上游源码参考） | main 分支 | 算法与测试的语义来源 | **不作为安装依赖**，按需最小化 vendor |
| einops | — | 上游仅反向 GQA 归约用到 `reduce` | 适配时以 torch 原生 `view/sum` 替代，去除依赖 |

### 2.2 内部适配模块（fla 公共件的最小化 vendor 清单）

验收要求 `tests/ops/test_wall_attn.py` 的断言与容差不得修改。该测试文件与上游 fla 仓同路径同内容，其 import 为 `fla.ops.wall_attn`、`fla.ops.utils.*`、`fla.utils`。为保持测试文件**逐字节不变**，方案 A（推荐）为在任务工作区 vendor 一个最小 `fla` 命名空间子集，使 import 路径与上游一致：

| 模块 | 内容 | 适配动作 |
|------|------|----------|
| `fla/ops/wall_attn/{__init__,parallel,decode,naive}.py` | 算子主体 | 移植 + NPU 适配（本文档第三章） |
| `fla/ops/utils/constant.py` | `RCP_LN2` | 原样 vendor |
| `fla/ops/utils/op.py` | `exp2`/`log2`（`@triton.jit` 小函数） | 原样 vendor，验证 NPU 后端 lowering |
| `fla/ops/utils/cumsum.py` | `chunk_global_cumsum`（scalar/vector 两个扫描 kernel） | 移植；NPU 上验证 `tl.cumsum(reverse=True)` |
| `fla/ops/utils/__init__.py` 中 `prepare_chunk_indices` | varlen 块映射 | 原样 vendor（host 侧 torch 实现） |
| `fla/utils.py` 子集 | `contiguous`、`autocast_custom_fwd/bwd`、`autotune_cache_kwargs`、`assert_close`、`device`、`input_guard` | 精简 vendor；`device` 指向 `npu` |
| `fla/utils.py` 中 `check_shared_mem` | GPU arch 判定（hopper/ampere） | **替换为 NPU SoC 判定**（torch_npu SoC Version / triton-ascend runtime 查询），驱动 block size 策略 |
| `fla/ops/backends` 的 `dispatch` | GPU 后端分发装饰器 | 移除，直接绑定 Triton 实现 |

### 2.3 需求模块设计

#### 2.3.1 算子原型

保持与上游完全一致的公开接口（语义、参数顺序、默认值、报错条件不变）：

```python
def parallel_wall_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,      # [B, T, HQ]
    sink_bias: torch.Tensor | None = None,     # [HQ]
    scale: float | None = None,                # 缺省 K**-0.5
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:                              # [B, T, HQ, V]，支持 autograd

def build_wall_kv_cache(
    k: torch.Tensor, g_cumsum: torch.Tensor, chunk_size: int,
    *, out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:         # (k_tilde, r_cache)

def parallel_wall_attn_decode(
    q: torch.Tensor, v: torch.Tensor, p_curr: torch.Tensor,
    k_tilde: torch.Tensor, r_cache: torch.Tensor,
    sink_bias: torch.Tensor | None, scale: float, cache_chunk_size: int,
    g_scalar_cumsum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:         # (o, lse)

def naive_wall_attn(...)                        # 精确参考，测试对拍用
```

#### 2.3.2 算子约束

| 约束 | 说明 |
|------|------|
| K ≤ 256 | 前向/assert `NK == 1`（单 K 块）；decode 同约束 |
| HQ % H == 0 | GQA 分组整除 |
| g ≤ 0 | 门控须为 log 衰减（logsigmoid 值域）；R 参考点分解依赖 P 单调不增，双侧门控不属于支持域（上游测试注释明确） |
| varlen 要求 B == 1 | 与上游一致；序列间隔离由 chunk_indices + bos/eos 保证 |
| 末维连续 | q/k/v/g_cumsum 等末维 stride == 1（host 侧校验 + contiguous） |
| dtype | fp16 / bf16 / fp32 输入；g/g_scalar/sink 参与 fp32 前缀和；softmax 累加 fp32 |
| 头维小块补齐 | K/V 非 2 幂或 < 16 时，BK/BV 取 `max(16, next_pow2(dim))` 掩码载入（满足 NPU `tl.dot` 最小维度与对齐要求；上游测试含 K=3/8/24、V=3/8/10 等小维） |

## 三、需求详细设计

### 3.1 使能方式

任务工作区布局（对齐社区任务目录规范，PR 时映射到 `tasklists/{编号}-wall_attn/{TeamName}/`）：

```
wall_attn_task/
├── docs/
│   └── design.md                      # 本文档
├── kernel/                            # vendor 的最小 fla 命名空间（PYTHONPATH 引入）
│   └── fla/
│       ├── utils.py                   # 精简工具集（device='npu'、assert_close 等）
│       └── ops/
│           ├── wall_attn/{__init__,parallel,decode,naive}.py
│           └── utils/{constant,op,cumsum,__init__}.py
└── tests/
    └── ops/
        └── test_wall_attn.py          # 与上游逐字节一致（断言与容差不改）
```

- 通过 `tests/conftest.py` 将 `kernel/` 注入 `sys.path`，使测试文件的 `import fla...` 命中 vendor 实现；
- 开发顺序：先跑通 naive 参考与 cumsum 工具（纯 torch 依赖少），再前向、再反向、再 decode；
- 若接口人对目录布局另有约定（如直接放 `src/triton_ascend_kernels/`），仅移动文件、不改实现与测试。

### 3.2 需求总体设计

#### 3.2.1 Host 侧设计

##### 3.2.1.1 分核策略

保持上游 grid 组织，并按 NPU 约束校验：

| kernel | grid | 说明 |
|--------|------|------|
| fwd / bwd_dq / bwd_dkv | `(NV, NT, B·HQ)` 3D | NT = cdiv(T, BT)；varlen 时 NT = len(chunk_indices)。含 `tl.dot`，按 AI Core 数组织，总 program 数与 A2/A3/950 核数匹配由 runtime 调度 |
| bwd_preprocess | `(NV·N,)` 1D | N = o.numel()/V |
| decode | `(NV·NT·B·HQ,)` 1D 展平 | 上游已为展平形式，天然适配 NPU |
| cumsum（工具） | `(cdiv(S,BS), N·H)` 2D | 沿用 fla 通用实现 |

约束处理：`coreDim ≤ 65535`（UINT16_MAX）——测试 shape 下各维远低于上限； Host 侧增加防御性检查，超大 shape 时给出明确报错而非越界行为。

##### 3.2.1.2 数据分块与内存（UB）预算策略

NPU UB 为 192KB（A2/A3），tile 选择必须使单 program 的活跃 buffer 总量受控。上游按 GPU shared memory 分档（hopper/ampere/其他），适配时替换为 **NPU SoC 分档**：

| 参数 | 上游（GPU） | NPU 适配策略 |
|------|------------|--------------|
| BK | hopper/ampere ≤ 256 | `min(256, max(16, next_pow2(K)))`，K ≤ 128 时单块；大 K 注意 fp32 g_cumsum 块的 UB 占用（BT×BK×4B） |
| BV | hopper ≤ 256 / ampere ≤ 128 / 其他 ≤ 64 | 默认 ≤ 64（A2），A3/950 可放宽至 128；满足 `BT·BV ≤ 16384` 的 fp32 累加器预算（继承上游剪枝规则） |
| BT / BS | autotune {64,128}×{32,64} | 收敛为 NPU 精简配置集（见 3.2.1.3） |
| num_warps / num_stages | autotune {2,4,8}×{2,3} | NPU 后端语义不同（对应核内流水/分核），以 triton-ascend 3.2.2 支持的行为为准实测收敛，不照搬 GPU 档 |

前向单 program 主要 buffer 估算（BT=64, BS=32, BK=128, BV=64，fp16 输入）：q 块 16KB + g_cumsum 块 fp32 32KB + k/v 块 8KB + 累加器（o: BT×BV fp32 16KB + 标量若干）≈ 80KB 量级，留有余量；反向 dq/dkv 的驻留 buffer 更多（q/k/v/do/P/R + dq/dg 累加器），以小配置为基线、逐平台实测上调。**UB overflow 是主要风险之一**（见第六章）。

##### 3.2.1.3 Autotune 与路径选择策略

- 定长路径：保留 autotune，但**配置集按 NPU 精简**（候选 BT∈{64,128}、BS∈{32,64} 的小子集 + 每平台 1–2 组 warps/stages），并保留 `BT·BV ≤ 16384` 剪枝；规避 issue 中"多配置 × IEEE fp32 dot 编译缓慢"表现为假死/超时的问题（上游注释明确指出该现象）；
- varlen 路径：沿用上游**固定 BT=128、绕过 autotuner**的策略（chunk 映射与 BT 绑定），BS 按 T 收敛到 16–64；
- `TRITON_F32_DEFAULT=ieee`：测试文件强制 IEEE fp32 dot（Wall 的 log 域分解对小门的灾难性抵消敏感，TF32 会破坏 dg 精度）。需验证 triton-ascend 3.2.2 对该环境变量（或等价 `input_precision`）的支持；若后端忽略该设置，在 kernel 层面对 fp32 输入显式指定 ieee 精度，**不得为提速偷换 tf32**（精度验收红线）；
- `WALL_ATTN_DKV_DIAG_BF16` 环境开关保留（反向对角块 bf16 快路径 / fp32 精路径），fp16 输入时该开关默认行为与上游一致。

#### 3.2.2 Kernel 侧设计

##### 3.2.2.1 逐 kernel 适配要点

| kernel | 计算要点 | NPU 适配关注点 |
|--------|----------|----------------|
| cumsum scalar/vector | 块内 `tl.cumsum`（含 reverse）+ 跨块运行值 | 验证 `tl.cumsum(axis=0, reverse=True)` 在 NPU 后端的支持；扫描类属于非连续规约，属 triton-ascend 已知能力边界区，必要时退化为 torch 前缀和（正确性优先）再优化 |
| fwd | off-diag（块参考 R，低精度 dot）+ diag（子块局部参考 + clamp 110 + 因果/窗口掩码）+ online softmax + sink 项 | ① `tl.dot` 最小维度 ≥16，小 K/V 掩码补齐；② 循环边界为运行期标量（`min((i_t+1)·BT, T)`），验证 NPU 控制流支持；③ `tl.trans`/`broadcast_to` pattern；④ int64 地址算术保持但消重；⑤ 指数/对数映射到向量指令 |
| bwd_preprocess | δ = rowsum(o∘do) | 低风险，向量 kernel |
| bwd_dq | 两循环（off-diag 累 dq_til、diag 直累输出域）+ 结构性 dg = ln2·q⊙dq | ① `tl.dot(a, b, acc)` 累加形式支持性；② fp32 ieee dot；③ `tl.where` 掩码与分支内 load 的 NPU 限制（FAQ：控制流与 load/store 组合支持不完备——保持上游"循环内直接 load"的写法，不做指针合并优化） |
| bwd_dkv | 对角循环（DIAG_BF16 双路径）+ off-diag 循环，dv/dk/dg | 同上；另注意 `b_ds.to(b_q.dtype)` 的降精度 dot 输入 |
| decode | 按 cache chunk（BS=C）循环，锚点 r_cache rescale | C ∈ {32,64,128}，BS=C 与 UB 预算联动校验；1D 展平 grid 已适配 |

##### 3.2.2.2 精度设计（不可退让项）

1. softmax 统计量（m、acc、lse、δ）全程 fp32；
2. off-diag 的 q_til/k_til 降至输入 dtype 走 Cube——安全性由"P 单调不增 ⇒ 指数 ≤ 0"保证（g ≤ 0 是接口契约）；
3. diag 子块局部参考 + 指数 clamp 110，防 fp32 溢出（inf·0=NaN 污染梯度）；
4. 测试环境 `TRITON_F32_DEFAULT=ieee` 下，fp32 输入的 dot 必须真 IEEE；
5. dg 链路：核内结构性导出 + host 侧 reverse cumsum（乘 RCP_LN2），与上游逐行对应；
6. 容差沿用上游（见 5.1），**不改断言、不改容差、不 skip 用例**。

##### 3.2.2.3 与上游实现的差异分析（适配改动清单）

| 类别 | 改动 | 理由 |
|------|------|------|
| 后端分发 | 移除 `fla.ops.backends.dispatch` 装饰器 | GPU 多后端机制，NPU 场景无意义 |
| 平台判定 | `check_shared_mem('hopper'/'ampere')` → NPU SoC 档位（torch_npu SoC 查询） | 驱动 BK/BV/BS 与 warps 选择；注意上游测试会 monkeypatch 该函数（`fla.ops.wall_attn.parallel.check_shared_mem`），**替换实现必须保持同名同模块位置可打桩**，否则测试失效 |
| 依赖去除 | einops.reduce → torch 原生分组求和 | 减少依赖 |
| autotune | 配置集 NPU 精简；保留 key/prune 语义 | 规避编译假死与 507014/507034 超时 |
| device | `fla.utils.device` → `'npu'`，引入 torch_npu | 运行环境 |
| 防御性检查 | grid 维度 ≤ 65535、UB 预算断言 | NPU 约束 |

kernel 算法本体（循环结构、掩码、online softmax、R 分解、clamp、结构性 dg）**不做语义改动**——这是精度对拍能过的前提。

### 3.3 支持硬件

Ascend A2 / A3 / 950（Atlas 训练与推理系列对应 SoC）。平台差异通过 SoC 档位参数（BK/BV/BS/warps）吸收，算法路径一致；三平台均需全量通过验收测试。

### 3.4 算子约束限制

- K ≤ 256，BK 单块；V 可超 BV（NV 多切 + host 归约）；
- 输入末维连续、ND 格式；元素总数 > 0；
- g 值域 ≤ 0（log 衰减门）；varlen 仅支持 B == 1；HQ % H == 0；
- fp32 输入走 IEEE dot（性能低于 bf16 属预期，正确性优先）。

## 四、特性交叉分析

算子的可选特性由 4 个 constexpr 开关（USE_SINK_BIAS / USE_WINDOW / IS_VARLEN / USE_SCALAR_G）经 heuristics 推导，**每种组合独立编译实例化**，无运行期相互污染。交叉矩阵与上游测试覆盖的对应关系：

| 维度 | 取值 | 覆盖测试 |
|------|------|----------|
| dtype | fp32（全路径）/ bf16（decode）/ fp16（验收补充） | test_parallel_*、test_decode_* |
| window | None / 8 | test_parallel_matches_reference 参数化 |
| varlen | 定长 / cu_seqlens（2 段） | test_parallel_varlen_matches_reference、test_backward_value_split（varlen+window+sink+scalar 全叠） |
| GQA | MHA / G=4 | test_parallel_gqa_matches_reference、decode 参数化 |
| V 切分 | NV=1 / NV=2（V=96/128/320） | test_backward_value_split_matches_single_tile、test_decode（V=320） |
| sink / scalar gate | 开 / 关 | test_parallel_sink_bias、test_scalar_gate_*、decode scalar gate |
| 长序列 | T=512 强衰减 / T=4096 decode | test_parallel_aggressive_gates_long_seq、test_decode_long |

varlen 不串扰由 chunk_indices→bos/eos 的段内掩码保证（naive 参考同样按段掩码，对拍有效）。交叉组合爆炸的风险由"constexpr 独立实例 + 以全叠组合测试（test_backward_value_split）做冒烟"控制。

## 五、可维可测分析

### 5.1 精度标准与测试方案

验收测试即上游 `tests/ops/test_wall_attn.py`（12 个测试函数，容差为文件内常量，不得修改）：

| 容差 | 值 | 适用 |
|------|----|------|
| RTOL_FWD | 5e-3 | Triton 前向 vs naive 精确参考（fp32） |
| RTOL_GRAD | 5e-3 | dq/dk/dv vs eager autograd |
| RTOL_FD | 2e-2 | dg、dg_scalar 有限差分 |
| RTOL_DECODE | 2e-2 | decode vs 训练前向自一致（fp32/bf16） |

测试清单：前向 3 组参数化（含 window）、GQA、varlen、sink、强衰减长序列、反向 2 组参数化、反向 V 切分一致性、dg 非零有限性、dg/dg_scalar 有限差分、scalar gate 前向 2 组、decode 4 组参数化 ×{fp32,bf16}、decode 长序列、decode scalar gate、decode 流式 serving 端到端、cache 布局形状。

调试手段：保留 `WALL_ATTN_DEBUG=1`（全链路 isfinite 断言）与 `WALL_ATTN_DKV_DIAG_BF16` 开关；问题定位顺序：naive（NPU torch）→ cumsum → fwd 定长 → fwd varlen → bwd → decode，逐层二分。

### 5.2 性能标准

任务以功能/精度验收为主，性能要求为"符合 Triton-Ascend 编程规范并针对 NPU 优化"：① 前向/反向/decode 显著优于 naive eager 实现；② 相对上游 GPU 实现无数量级劣化；③ 在 benchmark/ 目录补充 do_bench 数据作为 PR 附件（不作硬性门槛）。

### 5.3 兼容性分析

- 前向兼容 fp16/bf16/fp32 × 全部特性开关组合；接口与上游一致，可直接替换 fla 对应模块；
- 不修改 triton-ascend 编译器本体与 Kernels 仓既有算子，无回归面；
- 若适配中发现 triton-ascend 3.2.2 后端缺陷（如 cumsum reverse、ieee dot），以 issue 反馈 + kernel 侧规避双线推进，在 PR 根因分析中说明。

## 六、风险分析与对策（对应验收第 3 条）

| 风险 | 表现 | 根因假设 | 对策 |
|------|------|----------|------|
| aicore 超时 507014 | kernel 执行超 watchdog | 单 program 串行循环过长（小 BT × 大 T）；或 autotune 逐配置实测时长路径 | 精简 autotune 配置集；保证 BT≥64；varlen 固定 BT=128；超大 T 的 diag 循环界收窄 |
| vector core 超时 507034 | 同上（向量侧） | exp2/where 等大 tile 向量链过长 | 缩小 BS/BV；检查 clamp/掩码链是否可被 CSE |
| 编译假死 | autotune 阶段数分钟无输出 | IEEE fp32 dot 编译慢 × 配置数多（上游注释已记录该现象） | 配置集精简 + autotune 缓存（autotune_cache_kwargs 保留） |
| 编译器崩溃 | TritonToLinalg/HIVM 等 pass 报错 | `tl.trans`/`broadcast_to`/多 buffer 复杂 pattern 触发后端缺陷 | 最小复现 → 反馈 triton-ascend；kernel 侧以等价 pattern（expand_dims+broadcast）规避 |
| UB overflow | 编译期 ub overflow 报错 | tile 超出 192KB 预算 | 按 3.2.1.2 预算表收敛配置；优先降 BS/BV |
| ieee dot 不支持 | fp32 精度不达标 | 后端忽略 TRITON_F32_DEFAULT | kernel 显式 input_precision；若 cube 不支持则评估 vector fp32 dot 路径，全程留精度日志 |
| `tl.cumsum(reverse=True)` | cumsum 工具 kernel 失败 | 扫描类支持边界 | torch 前缀和兜底（host 侧，正确性优先），性能后续优化 |
| int64 地址开销 | 性能下降/编译告警 | 上游全 int64 寻址 | 保持语义，消重基址计算；必要时对小于 2^31 元素的场景用 int32 局部偏移 |
| `tl.dot(a,b,acc)` 累加形式 | bwd 编译失败 | 后端支持性待验证 | 改写为 `acc += tl.dot(a,b)` 等价形式 |

## 七、里程碑计划

| 阶段 | 内容 | 出口标准 |
|------|------|----------|
| M0（本周） | 设计文档评审 | 评论"申请文档验收" |
| M1（第 1–2 周） | 环境搭建（CANN 9.1.0 + torch_npu 2.9.0 + triton-ascend 3.2.2）；vendor 骨架；naive + cumsum 在 NPU 跑通；fwd 定长 fp32 过测 | test_parallel_matches_reference 通过 |
| M2（第 3–4 周） | fwd 全特性（window/varlen/sink/scalar）+ bwd 全链路 + decode；fp16/bf16 | 12 项测试全过（单平台） |
| M3（第 5–6 周） | A2/A3/950 三平台验证；超时/崩溃问题根因与收敛；性能初调；PR 附根因分析与验证结果 | 评论"申请验收"，向 gitcode Ascend/triton-ascend-kernels(experimental) 提 PR |
