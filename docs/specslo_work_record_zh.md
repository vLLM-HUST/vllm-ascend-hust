# vLLM-Ascend-SpecSLO 工作记录

更新时间：2026-09-08

## 1. 项目范围

本项目的目标是在昇腾 NPU 上实现论文和参考实现中的 SpecSLO 思路，
并复用此前 nano-PEARL 迁移中形成的跨模型投机解码、KV cache、图执行和
性能分析基础设施。代码以 `vllm-ascend-hust` 为主仓库；vLLM 核心调度器
的配套改动保存在 `patches/vllm-hust/`，因为核心仓库和 Ascend 插件是两个
独立的 Python 包，不能简单地把两个 Git 历史合并成一个安装包。

参考材料：

- ATC 论文：`/root/data/atc26-paper1664.pdf`。
- 参考代码：`rzwang22/atc26v0`，复现时固定在提交 `6b1cebf`。
- 相关上游：nano-PEARL 及 vLLM/vLLM-Ascend 的对应实现。

论文中的核心机制是双批次重叠执行：一批请求由 draft 推进，另一批请求由
target 验证；调度器按照 SLO 紧迫度、接受率和设备 roofline 预算动态选择
继续 draft、交给 target 验证或回退到自回归路径。验证成功时提升已接受的
前缀，失败时只保留接受前缀并使被拒绝后缀失效，同时允许新请求进入空出的
位置。

## 2. 已完成的迁移工作

### 2.1 nano-PEARL 原生运行时

- 实现 `PEARLConfig`、`PEARLEngine`、采样参数、请求队列和 `generate`、
  `AR_generate`、`bench_generate` 接口。
- draft/target worker 在同一个 HCCL world 中启动，使用独立的模型组和
  verification/correction 通信组，支持 `1TP + 2/3TP` 等异构 TP 布局。
- 支持 Qwen2、Qwen3、Llama 的模型加载和 safetensors 权重路径。
- target logits 在比较和采样前按 draft vocabulary 裁剪，覆盖 Qwen2.5
  0.5B/14B 这类词表大小不同的组合。
- 完成 pre-verify、gamma draft、target verify、rollback/correction 的
  原生执行链路，并记录每个请求的接受 token 数和 target 修正 token。

### 2.2 Ascend/NPU 适配

- 使用 HCCL 替代 NCCL，统一 world、rank、device 和 subgroup 拓扑，加入
  proposal id、request id、home rank、epoch、width、confidence 的自描述
  envelope，避免旧消息、错位请求和变宽 proposal 被误消费。
- 使用 CANN fused-infer attention、paged attention、FIA 和
  `_npu_reshape_and_cache` 写入 KV；KV page 固定为昇腾后端要求的 128 token，
  支持懒分配、完整 page 前缀复用、slot/block table 和请求回收。
- 使用 `torch.npu.NPUGraph`/ACLGraph；统一 draft shape，target 验证行数按
  bucket 规范化，图捕获达到 `max_aclgraph_entries` 后对新形状回退 eager。
- 对 Qwen3 BF16 标准 RoPE 自动复用生产 `qkv_rmsnorm_rope`，Qwen2/Llama
  unfused 路径复用生产 `npu_rotary_embedding`，不支持的架构保留 fallback。
- 针对 TP3 增加任务队列、HCCL AIV expansion 和确定性归约的可控默认值；
  保留 TP2/TP4 的显式环境变量覆盖。
- 增加 CPU/NUMA/IRQ affinity、prefill chunk、连续批处理、完成行 padding、
  full/half graph bucket 和 SLO/分阶段 profiling 统计。
- 保留 TP3 MC2、融合 FFN、QKV NZ、权重预取、通信 overlap 等实验入口，便于
  后续在真实 CANN 版本上替换成经过验证的 kernel。

### 2.3 vLLM 核心层配套

- 在 vLLM V1 增加 tree verification、SpecRhythm 元数据、词表映射、请求
  字段、调度器输出和 speculative 配置字段。
- 对通用 vLLM 路径增加树结构和节奏控制的数据结构及单元测试；真正的跨
  draft/target HCCL 执行仍由 Ascend 原生 PEARL engine 负责。
- 这些改动以可审阅补丁形式放在 `patches/vllm-hust/`，应用目标是
  `/root/data/vllm-hust` 对应的 `codex/nano-pearl-ascend-migration` 分支。

### 2.4 文档、示例和测试

- 提供 native target-only、native speculative、串行 speculative、连续批处理、
  profiling、TP3 collectives/MC2 等 benchmark 示例。
- 记录 graph bucket、paged KV、生产 RoPE、prefill chunk、warmup 和
  `elapsed_time`（包含 prefill + generation）的实验语义。
- 增加 PEARL、SpecRhythm、树 KV、树设备索引、ACLGraph、EAGLE proposer、
  vocab crop 和 AscendC 配置测试。

## 3. 无法直接从 CUDA/vLLM 搬运的部分

下面这些不是简单改 import 或后端名称就能完成的迁移点，因此做了专门设计。

| CUDA/上游假设 | 昇腾上的问题 | 当前设计 |
| --- | --- | --- |
| CUDA Graph、FlashAttention、Triton KV 写入 | CUDA kernel、stream/event 和 graph update API 与 CANN 不兼容 | NPUGraph/ACLGraph + FIA/PA + CANN cache op，并按 shape bucket 捕获 |
| NCCL 和跨 engine CUDA stream | HCCL rank/group、通信时序和设备可见性不同 | 单一 HCCL world、显式 subgroup、带 epoch 的 envelope 和同步点 |
| GPU block table/KV layout | vLLM-Ascend 的 page 和 slot 约束不同，page size 不是任意值 | 128-token page pool、prefix reuse、请求级 table 适配不同 attention backend |
| 一个 vLLM scheduler 直接拥有两套 worker | 通用 V1 worker 没有跨模型 HCCL group 边界 | 通用路径只传 metadata；可执行的双模型流程放在 native PEARL engine |
| 任意 TP 和均匀 head 切分 | Q/KV head、MLP、词表不能被 3 整除时 shape 不合法或通信浪费 | 参数/head/MLP/vocab padding、逻辑行裁剪；接受率和 padding 成本需单独评估 |
| 动态 shape graph | CANN graph capture 需要稳定 shape，频繁 capture 会耗尽内存 | 逻辑 proposal width 与物理 graph shape 分离，target 行数 bucket 化，超限 eager fallback |
| CUDA 异步输出和 KV 提交 | 拒绝后缀可能已写入设备 cache，旧结果可能晚到 | proposal/epoch 校验、commit/rollback 边界、失效后缀清理和状态快照 |
| PEARL-2 训练/蒸馏流水线 | 参考仓库仅有静态训练接口，缺少 Ascend 训练 kernel/数据管线 | 已补齐 teacher rollout、JSONL trace、acceptance-weighted loss、训练 step 与 checkpoint；真实训练质量仍需验证 |

## 4. 已完成的自研设计

1. **原生双模型拓扑**：在一个进程组内隔离 draft/target TP，统一设备发现、
   HCCL 初始化、verification/correction group 和 rank 映射。
2. **自描述通信协议**：每轮携带 request、proposal、epoch、逻辑宽度和置信度，
   接收端拒绝过期、错请求和不匹配宽度，解决异步 HCCL 下的状态污染。
3. **SpecRhythm 控制面**：记录接受率 EMA、SLO urgency、draft/verify 成本，
   按预算和队列状态选择 gamma、继续 draft 或 target 验证。
4. **设备侧树基础设施**：实现树节点索引、祖先路径、唯一 KV 位置映射、native
   target tree forward、显式 ancestor mask、验证结果回写和接受路径 KV compaction
   plan；通用 V1 tree verification 已有单元测试。
5. **Ascend graph/cache 适配**：物理 graph bucket、128-token page、完成行
   padding、prefix reuse、生产 RoPE 和 fused attention fallback 已接入 native
   runtime。
6. **动态离线批处理**：prefill queue、chunked prefill、连续 admission、请求
   完成替换和 SLO/goodput 统计已实现，适合有限 GSM8K/ShareGPT workload。

## 5. 仍需设计或验证的内容

- **TP3 MC2 的实机验证**：custom AscendC MC2、meta、编译 fusion pass、communicator
  解析和 `pearl/mc2.py` dispatch/fallback 已完成；仍没有在目标 CANN/固件版本上
  证明稳定地超过生产 all-reduce + matmul。
- **SpecRhythm 与树状投机解码的实机回归**：策略、native target forward、唯一
  cache position、KV compaction plan 和 rejection-safe 数据结构已完成；仍需在真实
  Qwen/Llama 模型上压测分支提交、回滚和数值一致性。
- **通用 vLLM 服务路径**：`SpecRhythmScheduler` 与
  `PearlDualModelScheduler` 已提供 admission、双 batch 并行回调、preempt/reactivate、
  global roofline 和 verification commit；仍需绑定上游 V1 scheduler 的 worker 生命周期。
- **PEARL-2 训练/蒸馏**：`pearl/distill.py` 已提供 teacher rollout、JSONL trace
  loader/collator、acceptance-weighted KL/CE、梯度裁剪、optimizer step 和 checkpoint；
  仍需大规模训练、teacher/student 权重产出和质量回归。
- **生产级动态 shape graph（运行时 guard 已完成，版本矩阵待验证）**：native
  graph 已按 shape bucket 捕获、回放、首轮 eager 对照和容量 fallback；仍需按
  CANN 版本和真实到达分布建立 capture/replay 兼容矩阵。
- **性能和稳定性回归**：需在固定 NPU 型号、驱动/CANN、模型量化配置下重新测
  batch、gamma、接受率、端到端延迟、SLO goodput、长上下文和多租户抢占。
- **扩展覆盖面**：多模态、LoRA、structured output、更多 tokenizer/vocab 映射
  以及异常退出后的 HCCL 资源回收仍需补齐。

## 6. 验证记录

本次迁移收尾执行了以下检查：

- Ascend 重点单元测试：`223 passed, 13 skipped, 14 warnings`。
- bridge/vocab 单元测试：`28 passed, 14 warnings`。
- vLLM 核心 tree/SpecRhythm 测试：`10 passed, 14 warnings`。
- `compileall` 和 `git diff --check` 均通过。
- NPU smoke：Qwen2.5 0.5B + 14B、TP1+TP3、SpecRhythm 单请求和动态 admission
  均完成生成，并观察到 eager promotion/接受计数。
- 参考 atc26v0 测试：`109 passed, 1 skipped, 1 failed`。唯一失败项是
  `test_real_probe_only_fields_are_not_seeded_in_base_trace_defaults`，原因是
  参考仓库当前测试期望与其 trace 默认字段实现不一致；该文件未被本次 Ascend
  迁移修改，也不影响 NPU 推理运行时。

本轮收尾新增：

- SpecRhythm roofline、PEARL-2 distillation（含 JSONL loader/collator）、tree
  coordinator、scheduler 和 MC2 fallback 单元测试：相关回归合计 `140 passed`。
- `examples/check_specslo_capabilities.py --device cpu --tp-size 3` 可运行并输出
  JSON 能力矩阵；在 NPU 上会额外报告 ACLGraph、FIA、paged attention、RoPE 和
  MC2 custom op 的导出状态。
- `compileall` 和 `git diff --check` 通过。

本次继续实现并回归：

- native tree target forward：每个分支使用唯一 cache position，显式 ancestor mask
  强制走 dense correctness path；增加设备侧 batch verifier、bonus token 分离和
  接受路径 KV compaction/rollback API。
- 通用 `PearlDualModelScheduler`：基于 SpecRhythm schedule 并行提交 draft/target
  worker 回调，并提供 verification commit 生命周期入口。
- PEARL-2 teacher rollout：支持批量 prompt、temperature、EOS 截断、padding mask，
  以及 `collect_pearl_teacher_trace.py` JSONL CLI。
- TP residual MC2：增加 HCCL communicator 名称兼容探测、`enable_mc2` 配置和
  fused-op 失败自动 fallback；默认关闭，需在目标 CANN 上显式开启验证。
- 本轮相关单测为 `132 passed, 14 warnings`；完整 `tests/ut/spec_decode` 还受容器
  缺少 `numba` 影响，ngram proposer 收集失败，其余可收集用例通过。

参考仓库逐文件审计见
[`atc26v0_feature_audit_zh.md`](atc26v0_feature_audit_zh.md)。该审计把上游
scaffold/probe 与真正可执行的 nano-PEARL 功能分开记录，避免把探针通过误报为
生产迁移完成。

本次只做功能迁移和验证，没有宣称新的吞吐提升；后续性能报告必须注明模型、
TP、batch、gamma、warmup、CANN/驱动版本和端到端计时口径。
