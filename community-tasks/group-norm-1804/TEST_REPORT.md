# Liger GroupNorm Ascend 适配与性能优化 — 测试报告

> 社区任务：[triton-lang/triton-ascend#1804](https://github.com/triton-lang/triton-ascend/issues/1804)
> 算子：LigerGroupNorm（前向 + 反向），上游参考 liger_kernel v0.8.2
> 报告日期：2026-09-17；测试人：ZCode（hanggongmao 团队任务）
> 设计文档：`community-tasks/group-norm-1804/design.md`（第八章 As-Built 与本文数据一致）

---

## 一、测试环境

| 项 | 版本 / 配置 |
|---|---|
| 硬件 | Ascend 950PR（9579，131GB HBM），x86_64 openEuler |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu（cp312，download.pytorch.org CPU 源） |
| torch_npu | 2.9.0.post8（**注意：950PR 必须 ≥ 2.9.0.post2**，2.9.0 初版报 `Unsupported soc version: Ascend950PR 9579`） |
| triton-ascend | 3.2.2（GitHub release wheel `triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64`；pypi 与华为云镜像均无 3.2.2） |
| Python | 3.12.13 |
| liger_kernel | 0.8.2 语义（任务工作区 vendor 最小子集，见 design.md 2.2 节） |

## 二、功能与精度测试

### 2.1 验收测试（`pytest tests/test_group_norm.py`）

测试文件与上游 liger_kernel v0.8.2 `test/transformers/test_group_norm.py` 逐字节一致，**未修改任何断言与容差**（fp32，atol=rtol=1e-4）。

```
tests/test_group_norm.py::test_liger_group_norm[...-1-1-1-3]      PASSED   # 极小
tests/test_group_norm.py::test_liger_group_norm[...-1-32-32-4]    PASSED   # G == C（InstanceNorm 退化）
tests/test_group_norm.py::test_liger_group_norm[...-16-32-1-4096] PASSED   # G == 1（LayerNorm 退化）
tests/test_group_norm.py::test_liger_group_norm[...-2-63-21-2163] PASSED   # 非对齐 C/H
tests/test_group_norm.py::test_liger_group_norm[...-16-48-12-8192] PASSED  # 大 shape 多 tile
============================== 5 passed in 30.53s ==============================
```

**0 failed、0 error**，覆盖 Issue 要求的全部边界场景（num_groups==num_channels、num_groups==1、非对齐 hidden size）。前向 out、反向 dx/dw/db 四项断言均通过。

### 2.2 内部 dtype 扩展验证（非验收件）

`tests/internal_dtype_check.py`：fp32 / bf16 / fp16 × 6 组 shape（验收 5 组 + (4,64,16,512)），以 fp32 oracle 对照，判定准则为"liger 误差 ≤ max(torch 同 dtype 自身误差 × 2, dtype 地板值)"（bf16 输出 1-ulp 舍入差异属固有，torch 自身也存在）。

```
fp32 × 6 shapes：全部 OK（另按验收标准与 torch 直接对比 1e-4 通过）
bf16 × 6 shapes：全部 OK
fp16 × 6 shapes：全部 OK
ALL DTYPE CHECKS PASSED
```

加固措施（对上游的偏差，见 design.md 8.1）：Mean/RSTD/DW/DB 内部 buffer 改 fp32（上游按输入 dtype，bf16 下曾实测 dx 偏差 0.0156 超内部阈值）；统计与归约全程 fp32；`var = max(E[X²]−m², 0)` 防 rsqrt NaN。

## 三、性能测试

基准：`torch.nn.GroupNorm`（torch_npu 融合算子）。指标：`benchmark_group_norm.py` speed 模式 y_value_50（ms），ratio = liger / torch，**目标全配置 ratio ≤ 1.0**。原始日志见 `benchmark/results/`。

### 3.1 token_length sweep（llama_3_8b，bf16，H=512，channels_per_group=4）

优化前后对比（优化前 = 上游移植 + 基本 NPU 适配版）：

| BT | mode | torch (ms) | 优化前 (ms) | 优化前 ratio | 优化后 (ms) | 优化后 ratio |
|----|------|-----------|------------|-------------|------------|-------------|
| 1024 | forward | 0.0231 | 0.1226 | 5.421 | 0.0218 | **0.945** |
| 1024 | backward | 0.0676 | 0.2236 | 3.354 | 0.0440 | **0.672** |
| 1024 | full | 0.1362 | 0.3721 | 2.737 | 0.1237 | **0.911** |
| 2048 | forward | 0.0423 | 0.2268 | 5.528 | 0.0407 | **0.994** |
| 2048 | backward | 0.1138 | 0.4299 | 3.831 | 0.0801 | **0.732** |
| 2048 | full | 0.2453 | 0.7203 | 2.965 | 0.2290 | **0.943** |
| 4096 | forward | 0.0806 | 0.4287 | 5.318 | 0.0810 | **1.005** |
| 4096 | backward | 0.2051 | 0.8491 | 4.150 | 0.1515 | **0.735** |
| 4096 | full | 0.4661 | 1.4175 | 3.041 | 0.4409 | **0.942** |
| 8192 | forward | 0.1546 | 0.8371 | 5.619 | 0.1524 | **1.021** |
| 8192 | backward | 0.3576 | 1.6717 | 4.677 | 0.2954 | **0.826** |
| 8192 | full | 0.8972 | 2.8661 | 3.197 | 0.8517 | **0.951** |

复跑第二轮 forward：0.919 / 0.940 / 1.018 / 1.023；backward 与 full 各配置均稳定 < 1.0。

### 3.2 model_config sweep（BT=2048，bf16，优化后）

| 模型 | hidden | forward | backward | full |
|------|--------|---------|----------|------|
| deepseek_v2_lite | 2048 | 0.986 | 0.688 | 0.920 |
| deepseek_v3 | 7168 | 0.995 | 0.688 | 0.914 |
| llama_2_7b | 4096 | 1.018 | 0.735 | 0.959 |
| llama_3_8b | 4096 | 1.023 | 0.735 | 0.948 |
| qwen2.5_7b | 3584 | 0.881 | 0.644 | 0.912 |
| qwen2.5_14b | 5120 | 1.005 | 0.710 | 0.930 |
| qwen2.5_72b | 8192 | 1.006 | 0.715 | 0.928 |

### 3.3 显存（memory 模式）

| BT | liger (MB) | torch (MB) |
|----|-----------|-----------|
| 1024 | 56.1 | 56.1 |
| 2048 | 96.1 | 96.1 |
| 4096 | 176.1 | 176.2 |
| 8192 | 336.2 | 336.4 |

### 3.4 性能结论

- **backward 全配置 ratio 0.64–0.83**（最多比基线快 35%）；**full 全配置 0.91–0.96**；
- **forward 0.88–1.02**：与 torch 同处访存屋顶线。微基准实测（BT=8192）：纯拷贝 kernel 0.147ms、本算子 0.149–0.152ms、torch 0.147–0.155ms——三者同处硬件屋顶线，残余 ±3% 为基线测量噪声，继续压缩无物理空间；
- Benchmark 全部配置正常产出数据，无崩溃、无超时；显存占用与基线一致。

## 四、Profiling 根因分析（优化前后 2.7–5.6× → ≈1.0 的归因）

以最大配置（BT=8192，[16,4096,512] bf16，G=1024）控制变量微基准拆解：

| # | 根因 | 实测影响 | 优化手段 |
|---|------|----------|----------|
| 1 | 上游前向逐元素 `offset // H` + gather 载入 W/B | 0.23ms（占 44%） | 仿射参数改为 [GPB,CPG] 向量载入，reshape [GPB,CPG,H] 沿 H 广播；零整除、零 gather |
| 2 | 1D 全 tile `tl.sum` 归约（2048 元素/program） | 每次归约 0.05–0.10ms | **改 [GPB,NG] 2D tile + axis=1 行归约**：stats kernel 0.275ms → 0.103ms，单项最大收益 |
| 3 | 反向逐通道嵌套循环 + 每通道 2 次标量 atomic_add，X/dY 重复载入 | 反向 4.7× | reshape 分段归约（axis=2）+ batch 循环寄存器累加：零 atomic、零 zeros 初始化、零 cast kernel，反向仅 1 次 launch |
| 4 | 小配置下 triton python dispatch 18μs/次 > kernel 15μs | BT=1024 full ratio 1.34 | 缓存 CompiledKernel 直发（9μs）：full ratio 1.34 → 0.92 |
| 5 | 上游 NPU BLOCK=16384 fp32 编译期 UB overflow（需 640KB > 216KB） | 3/5 验收用例无法编译 | MAX_FUSED_SIZE 收敛至 4096（`LIGER_GN_MAX_FUSED_SIZE` 可调） |
| 6 | 标量 Mean/RSTD store、多 buffer 中间量 | 次要 | batch 循环摊薄 + 寄存器复用 |

Kernel 调度（host 按 shape 三分支）：NG 为 2 幂且 ≤4096 走融合 2D 热路径（benchmark 全配置命中）；NG ≤ BLOCK 非 2 幂走 1D 单 tile；NG > 4096 走多 tile 两遍（正确性路径，仅测试 shape 命中）。

## 五、复测命令

```bash
# 环境准备（950PR）
pip3 install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install "torch-npu==2.9.0.post8" pytest
pip3 install triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  # GitHub release

# 功能/精度
cd group_norm_task
PYTHONPATH=kernel python3 -m pytest tests/test_group_norm.py -v
PYTHONPATH=kernel python3 tests/internal_dtype_check.py

# 性能（产出 CSV/JSON 于 stdout，日志即 results/ 下文件）
cd benchmark
PYTHONPATH=../kernel python3 benchmark_group_norm.py --overwrite                            # token_length sweep
PYTHONPATH=../kernel python3 benchmark_group_norm.py --overwrite --sweep-mode model_config  # model_config sweep
python3 results/parse_bench.py <bench.log>   # ratio 汇总表
```

## 六、已知限制与后续工作

1. 前向大 BT 配置 ratio 在 1.0±0.03 波动（屋顶线噪声，见 3.4）；
2. 多 tile fallback（NG > 4096，benchmark 不覆盖）为正确性优先路径，未做深度性能优化；
3. NG 非 2 幂走 1D 单 tile 路径（性能介于两者之间）；
4. 本次实测平台为 950PR（Atlas A5，DAV_3510 架构）；**910B3（Atlas A2 系列芯片，与 A3 的 910_93 同属 DAV_2201 架构）及 A3 待复测**——融合路径全部为通用 Triton 语义，预期可直接运行，BLOCK 上限需按各平台 UB 容量复核（DAV_2201 UB 192KB，本机 DAV_3510 为 248KB，`LIGER_GN_MAX_FUSED_SIZE` 环境变量可调）；
5. 原始日志与解析脚本：`benchmark/results/`（baseline / opt1 / opt2 / r1 / r2 / mc 六份）。
