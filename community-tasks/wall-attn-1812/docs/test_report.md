# Wall Attention 算子 Ascend 适配 — 验收测试报告

> 任务：[triton-lang/triton-ascend#1812](https://github.com/triton-lang/triton-ascend/issues/1812)【社区任务】Wall Attention 算子 Ascend 适配
> 算子来源：上游 [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/ops/wall_attn`（Tilde Research 贡献）
> 设计文档：[design.md](design.md)　　代码：`kernel/fla/`（本目录上层）

## 1. 测试结论汇总

验收命令：`pytest tests/ops/test_wall_attn.py`（测试文件与上游 fla 仓**逐字节一致**，断言与容差未做任何修改）。

| 平台 | 状态 | 结果 | 备注 |
|------|------|------|------|
| Ascend 950（A5） | ✅ 已验证 | **31 passed, 0 failed, 0 error**（耗时 527s） | 实测机型 Ascend950PR (9579) |
| Ascend A2 | ⏳ 待验证 | — | 已安排其他同事在 A2 机器复测（见第 6 节部署要点，步骤可直接复用） |
| Ascend A3 | ⏳ 待验证 | — | 同上 |

测试覆盖：训练并行前向（定长/varlen/滑窗/sink bias/GQA/长序列强衰减）、训练反向（dq/dk/dv 对拍、V 维切分一致性、门控梯度有限差分）、scalar gate 前向与梯度、推理 decode（MHA/GQA/V 维切分/长上下文 4096/scalar gate/流式 serving 端到端/cache 布局）。数据类型覆盖 fp32 与 bf16（验收测试矩阵定义的全部参数化配置）。

## 2. 测试环境（950 平台实测）

| 组件 | 版本 |
|------|------|
| 硬件 | Ascend950PR (9579)，board A310-50-C00MM304A1，HBM 128GB，x86_64，openEuler 24.03 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post8（注意：2.9.0 初版不识别 950PR SoC，需 post8 及以上） |
| triton-ascend | 3.2.2（triton 3.2.0，需从 ascend 源安装，见第 6 节） |
| Python | 3.11.6（需 python3-devel 提供 kernel 编译所需 Python.h） |

## 3. 测试结果明细（950 平台，31/31 通过）

### 3.1 训练并行前向（10 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 1 | test_parallel_matches_reference[None-B1-T48-H2-HQ4-K32-V16] | ✅ PASSED |
| 2 | test_parallel_matches_reference[None-B2-T31-H1-HQ1-K24-V8] | ✅ PASSED |
| 3 | test_parallel_matches_reference[None-B1-T31-H1-HQ2-K32-V128] | ✅ PASSED |
| 4 | test_parallel_matches_reference[8-B1-T48-H2-HQ4-K32-V16]（滑窗） | ✅ PASSED |
| 5 | test_parallel_matches_reference[8-B2-T31-H1-HQ1-K24-V8]（滑窗） | ✅ PASSED |
| 6 | test_parallel_matches_reference[8-B1-T31-H1-HQ2-K32-V128]（滑窗） | ✅ PASSED |
| 7 | test_parallel_gqa_matches_reference（GQA G=4） | ✅ PASSED |
| 8 | test_parallel_varlen_matches_reference（varlen 双序列） | ✅ PASSED |
| 9 | test_parallel_sink_bias_matches_reference（sink bias） | ✅ PASSED |
| 10 | test_parallel_aggressive_gates_long_seq（T=512 强衰减） | ✅ PASSED |

### 3.2 训练反向（5 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 11 | test_backward_matches_eager_reference[B1-T24-H2-HQ4-K16-V12] | ✅ PASSED |
| 12 | test_backward_matches_eager_reference[B1-T64-H2-HQ2-K64-V128] | ✅ PASSED |
| 13 | test_backward_value_split_matches_single_tile（varlen+滑窗+sink+scalar 全叠，V 切分一致） | ✅ PASSED |
| 14 | test_dg_nonzero_after_backward（门控梯度有限性） | ✅ PASSED |
| 15 | test_g_gradient_matches_finite_differences（dg 有限差分） | ✅ PASSED |

### 3.3 Scalar Gate（3 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 16 | test_scalar_gate_matches_reference[B1-T48-H2-HQ4-K32-V16] | ✅ PASSED |
| 17 | test_scalar_gate_matches_reference[B2-T31-H1-HQ1-K24-V8] | ✅ PASSED |
| 18 | test_scalar_gate_gradient_finite_differences（dg_scalar 有限差分） | ✅ PASSED |

### 3.4 推理 Decode（13 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 19–20 | test_decode_matches_training_forward[B1-T256-H4-HQ4-K64-V64-C64]（fp32/bf16，MHA） | ✅ PASSED |
| 21–22 | test_decode_matches_training_forward[B1-T256-H2-HQ8-K64-V64-C64]（fp32/bf16，GQA G=4） | ✅ PASSED |
| 23–24 | test_decode_matches_training_forward[B2-T128-H1-HQ2-K32-V32-C32]（fp32/bf16） | ✅ PASSED |
| 25–26 | test_decode_matches_training_forward[B1-T128-H1-HQ2-K32-V320-C32]（fp32/bf16，V 维切分） | ✅ PASSED |
| 27 | test_decode_matches_training_forward_long（T=4096 bf16 长上下文稳定性） | ✅ PASSED |
| 28 | test_decode_with_scalar_gate | ✅ PASSED |
| 29–30 | test_decode_streaming_matches_full_forward[dtype0/dtype1]（流式 serving 端到端） | ✅ PASSED |
| 31 | test_decode_cache_layout_shapes（cache 布局形状） | ✅ PASSED |

精度容差（沿用上游测试文件内常量，未修改）：RTOL_FWD 5e-3、RTOL_GRAD 5e-3、RTOL_FD 2e-2、RTOL_DECODE 2e-2；fp32 dot 强制 IEEE（测试文件自带 `TRITON_F32_DEFAULT=ieee`）。

## 4. 问题与根因分析（适配过程中发现并修复）

**问题 1：头维非 2 的幂时 triton-ascend 3.2.2 后端编译失败（7 个用例）。**
现象：K/V 为 24/20/8/3 等非 2 幂维度时，前向 kernel 在 `buildFinalHIVMPipelines` 阶段报错 `vector.transfer_write op requires a permutation_map with result dims of the same rank as the vector type` 及 `tracking listener failed to find replacement op`。
根因：kernel 内对头维的掩码 load（`mask=(o_d < K)`）在 K≠BK 时产生的 permutation/transfer_write 模式触发后端缺陷；K==BK（2 的幂）时掩码恒真被折叠，路径正常。
修复：host 侧公开接口内将 q/k/v/g（decode 含 p_curr/k_tilde/r_cache）的 K/V 维零填充至 2 的幂（≥16），消除掩码；零填充对打分与输出无数学影响，梯度由 autograd 经 pad/slice 反向自动还原。该后端缺陷建议另行向 triton-ascend 反馈。

**问题 2：bwd_dkv kernel UB 溢出（1 个用例）。**
现象：varlen 反向（BT=128、num_stages=2）编译报 `ub overflow, requires 2972672 bits while 2031616 bits available`（950PR UB 253952B）。
根因：反向 dkv kernel 驻留 buffer 多（dv/dk 累加器、do/v/q 多块 tile、b_ds/b_p/b_dp 中间量），BT=128 + 双级流水超出 UB 预算。
修复：varlen 反向固定配置降为 BT=64、num_stages=1；反向 autotune 空间统一改为单级流水。

**问题 3：环境部署（非代码问题）。**
① pypi 官方源无 triton-ascend 3.2.2，需使用 ascend 源：`pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi`；② torch_npu 2.9.0 初版不识别 950PR（`Unsupported soc version: Ascend950PR 9579`），需 ≥ 2.9.0.post8；③ kernel 首次编译需 python3-devel（Python.h）。

## 5. 与任务验收标准对齐

| 验收标准 | 结论 |
|----------|------|
| `pytest tests/ops/test_wall_attn.py` 全量通过（0 failed、0 error），不改断言容差 | ✅ 950 平台 31/31 通过；测试文件与上游逐字节一致 |
| 覆盖 A2 / A3 / 950 | 950 ✅；A2/A3 待复测（代码无平台相关硬编码，block 配置按 UB 档位自适应） |
| 消除 507014/507034 超时及编译器崩溃 | ✅ 三轮全量运行（20+ 种 shape）无超时、无挂死；两处编译问题已根因修复 |
| 前向与反向梯度精度 | ✅ 全部容差内（含有限差分验证） |
| 定长+varlen、fp16/fp32 等参数化配置 | ✅ 验收矩阵（fp32/bf16 × 定长/varlen）全部通过 |
| 不引入回归 | ✅ 独立 vendor 目录，不改动 triton-ascend 编译器与任何既有算子 |

## 6. 复测部署要点（A2/A3 复用）

```bash
# 1. 安装依赖（openEuler/CentOS 系）
sudo dnf install -y python3-devel
pip install torch==2.9.0 torch_npu==2.9.0.post8 pytest numpy \
    -i https://repo.huaweicloud.com/repository/pypi/simple
pip install triton-ascend==3.2.2 -i https://repo.huaweicloud.com/repository/pypi/simple \
    --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
# 2. 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 3. 运行验收测试（kernel/fla 为 vendor 实现，tests/conftest.py 自动注入路径）
cd community-tasks/wall-attn-1812
python3 -m pytest tests/ops/test_wall_attn.py -v --tb=short
```

注意：A2 机器上 torch_npu 使用 2.9.0 即可（post8 为 950PR SoC 支持所需）；首次运行含全量 kernel 编译，约需 10–45 分钟。

## 7. 遗留事项

- A2 / A3 平台复测（已协调同事执行，复测步骤见第 6 节）；
- 性能调优（当前以功能/精度验收为目标；`ascend_compile_kwargs` 关闭了 auto-multi-buffer、autotune 空间精简过，后续可在保证稳定前提下放开并补 benchmark 数据）；
- 头维掩码编译缺陷建议形成最小复现，反馈 triton-ascend 社区。
