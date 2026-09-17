# Wall Attention 算子 Ascend 适配（社区任务 #1812）

任务 Issue：[triton-lang/triton-ascend#1812](https://github.com/triton-lang/triton-ascend/issues/1812)
设计文档：[docs/design.md](docs/design.md)

## 目录结构

```
wall-attn-1812/
├── docs/design.md            # 设计说明书（文档验收材料）
├── kernel/fla/               # 上游 fla 最小 vendor 子集（Ascend NPU 适配版）
│   ├── utils/                # device/assert_close/decorators/autotune 兼容层
│   └── ops/
│       ├── utils/            # RCP_LN2 / exp2·log2 / chunk_global_cumsum / prepare_chunk_indices
│       └── wall_attn/        # 算子本体：naive（参考）、parallel（训练 fwd/bwd）、decode（推理）
└── tests/
    ├── conftest.py           # 注入 kernel/ 到 sys.path；预导入 torch_npu
    └── ops/test_wall_attn.py # 验收测试，与上游 fla 仓逐字节一致（断言与容差不改）
```

## 运行方式

环境：CANN 9.1.0 + torch 2.9.0 + torch_npu 2.9.0 + triton-ascend 3.2.2。

```bash
cd community-tasks/wall-attn-1812
pytest tests/ops/test_wall_attn.py
```

## 与上游 fla 的差异（仅工程适配，数值语义不变）

1. 移除 `fla.ops.backends.dispatch`（GPU 后端路由）与 einops 依赖（GQA 归约改用原生 torch）。
2. `check_shared_mem` 在 NPU 上恒返回 False → 走保守 tile 路径（BV≤64、BS≤32）；
   函数名/签名与上游一致，上游测试的 monkeypatch 点不受影响。
3. autotune 空间精简（num_warps {2,4} × num_stages {2}），规避 NPU 上
   多配置 × IEEE fp32 dot 编译堆积（对应验收项 507014/507034 超时对策）。
4. kernel 启动附带 `ascend_compile_kwargs()`（关闭 auto-multi-buffer，控制 UB 占用）。
