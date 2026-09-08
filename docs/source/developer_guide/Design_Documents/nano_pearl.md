# nano-PEARL on Ascend

## Status

The native runtime is an Ascend port of the functionality implemented by
upstream nano-PEARL at commit `6b1cebf`. Draft and target ranks share one HCCL
world, own independent persistent KV caches, and execute the upstream
`pre-verify -> gamma draft -> target verify -> rollback` pipeline.

The primary API mirrors upstream nano-PEARL:

```python
from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams
```

`PEARLEngine` starts all draft and target workers with `spawn`, creates the
HCCL topology, accepts queued string or token-ID prompts, and exposes
`generate`, `AR_generate`, and `bench_generate`.

The repository also retains an OpenAI-compatible bridge. That bridge uses
vLLM's speculative scheduler and a separate draft service; it is useful for
serving, but is not the native cross-group PEARL pipeline.

## Feature Parity

| Upstream implemented feature | Ascend implementation |
| --- | --- |
| Qwen2, Qwen3, and Llama | Native TP models and Hugging Face safetensors loader |
| Independent draft and target TP | Disjoint HCCL model groups in one world |
| Dynamic TP 3, 6, and 7 | Upstream-compatible zero padding for heads, KV heads, MLP, and vocabulary |
| Request scheduling | Static chunks plus optional finite prefilled queue, bounded by batch and KV limits |
| Paged KV cache and prefix reuse | Shared, lazily allocated CANN page pool with full-page prefix caching |
| CUDA Graph | `torch.npu.NPUGraph` plus replay-time CANN FIA/PA graph-task metadata updates |
| FlashAttention and Triton KV write | CANN fused-infer attention, paged attention, and `_npu_reshape_and_cache` |
| Target temperature sampling | Exponential-race sampling and upstream stochastic verification rule |
| Per-request stopping | `max_tokens`, `ignore_eos`, and matching EOS validation |
| Automatic gamma | Startup profiles for batch buckets 1, 2, 4, 8, 16, and 32 |
| AR and fixed-step benchmarks | `AR_generate` and `bench_generate` |
| MAT reporting | Per-request `num_acc_tokens`, including target correction tokens |
| Offline continuous batching | Prefilled request queues with stable full/half ACLGraph batch buckets |

The target logits are cropped to the draft vocabulary before sampling or
comparison. This supports pairs such as Qwen2.5-0.5B-Instruct with a 151,936
entry model vocabulary and Qwen2.5-14B-Instruct with 152,064 entries, provided
the draft token IDs are an unchanged prefix of the target mapping.

Ascend uses 128-token KV pages. Therefore `kvcache_block_size` is present in
the compatible configuration surface but must be 128; this is the native
vLLM-Ascend paged-attention constraint rather than the upstream CUDA default.
`gpu_memory_utilization` or an explicit `num_kvcache_blocks` controls the
shared page-pool capacity.
Attention metadata constructs only the page-table layout consumed by the
selected backend: paged attention receives a table per packed token, while
fused-infer attention receives a table per request. The unused internal field
reuses the selected table instead of launching another device `index_select`.
`max_aclgraph_entries` defaults to 32 and bounds both graph-resident workspaces
and total capture attempts; unseen low-frequency shapes use eager execution
after the limit is reached, including when earlier captures were rejected.
Uniform draft shapes and canonicalized mixed target-verification shapes are
keyed independently. Target verification quantizes the post-verify row count
into `target_verification_graph_buckets` (8 by default), then discards padded
logit rows before verification. This bounds graph memory while keeping padding
outside the acceptance decision. On Ascend, eight buckets also avoid filling
the measured decode path with one ACLGraph capture for every exact mixed row
count; users can still override the value for a specific workload.

For offline workload tuning, `target_verification_graph_post_counts` can replace
the uniform target buckets for selected batch sizes. The benchmark exposes the
same setting as the repeatable option
`--target-verification-graph-post-counts BATCH:COUNT,COUNT,...`; counts must be
strictly increasing, start at zero, and stay within the configured batch size.
The default is empty, so normal runs continue to use uniform buckets. This is a
diagnostic and workload-specific control: different padded matrix shapes can
change BF16 greedy decisions, acceptance, and scheduling even when no graph
fallback occurs. Compare output hashes, acceptance, rounds, and end-to-end
latency before retaining a custom set.

`draft_use_paged_attention=True` selects CANN paged attention for the
one-token draft steps while retaining fused-infer attention for packed target
verification. It is disabled by default because the faster draft kernel only
improves end-to-end throughput when the target graph buckets keep the two
model groups balanced. The benchmark CLI exposes the same option as
`--draft-use-paged-attention`; tune it together with
`--target-verification-graph-buckets` and validate acceptance as well as
throughput.

For BF16 Qwen3 models with the standard 128-dimensional RoPE head, the native
runtime automatically reuses vLLM-Ascend's production
`vllm::qkv_rmsnorm_rope` Triton operator. It fuses QKV splitting, Q/K RMSNorm,
and rotary embedding; unsupported architectures, dtypes, head dimensions, and
RoPE variants keep the original operator sequence. The engine initializes the
Triton device properties after selecting the local NPU, matching the production
model runner. This path is automatic and has no separate CLI switch.

For the unfused Qwen2/Llama attention path, PEARL also defaults to
vLLM-Ascend's production `vllm::npu_rotary_embedding` operator instead of two
separate `npu_rotary_mul` launches. Draft and target selection is independent
through `draft_use_production_rope` and `target_use_production_rope`; both
default to `True`. The benchmark and direct worker CLI expose Boolean optional
flags, so `--no-draft-use-production-rope` or
`--no-target-use-production-rope` provides an explicit compatibility control.
The fallback remains available automatically when the production operator is
not registered.

Before creating the native HCCL world, PEARL defaults `TASK_QUEUE_ENABLE` to
`1` and `HCCL_OP_EXPANSION_MODE` to `AIV`. Queue mode 1 reduces host-side
operator submission overhead while retaining ACLGraph compatibility, and AIV
lets HCCL schedule communication on
the AI vector core. A TP3 target additionally defaults `HCCL_DETERMINISTIC` to
`true`; this selects CANN's deterministic AIV reduction for the sub-8 MiB
row-parallel tensors used by the validated Qwen2.5-14B target. All settings use
`setdefault`, so explicit user values are preserved and TP2/TP4 do not have
deterministic reduction forced on them. On the validated Qwen3-0.6B TP1 plus
Qwen3-32B TP2 workload, repeated B32 runs kept identical output hashes while
the AIV setting improved steady-state worker throughput.
`TASK_QUEUE_ENABLE=2` must not be combined with this ACLGraph runtime because
the current CANN release rejects that queue mode during NPU graph capture.

After all workers finish model and process-group initialization, PEARL applies
the same Ascend-native CPU, NUMA, and IRQ affinity policy as the production
vLLM-Ascend worker. `enable_cpu_binding` defaults to `True`; set it to `False`,
or pass `--disable-cpu-binding` to the runnable examples and benchmark, for an
explicit control run. Binding failures are logged and do not abort inference,
matching the production worker's fallback behavior.

`prefill_chunk_size` optionally limits the number of prompts packed into one
prefill forward without reducing `max_num_seqs`, the decode batch size. It
defaults to `max_num_seqs`. This is useful when prompt activations or a backend
operator limit cannot accommodate the full decode batch in one prefill.

With `enable_continuous_batching=True`, `max_num_seqs` remains the maximum
active decode batch while `max_num_queued_seqs` reserves KV pages for the
queued workload. Prompts are prefetched in `prefill_chunk_size`-sized chunks
before decode, completed rows are replaced by already-prefilled requests, and
the draining tail uses only full and half-size ACLGraph batches. This mode
targets finite offline workloads; it does not expose a live request-admission
server.

For fixed-size offline benchmarks, `pad_finished_requests=True` keeps completed
rows in the device batch until the last row finishes. Their externally visible
output and acceptance counters are frozen at completion, and each redundant
device round replays from that bounded completion snapshot while preserving the
full-batch ACLGraph shape. This option is disabled by default. Continuous mode
pads only to its full/half graph buckets, reducing tail waste while keeping
graph shapes bounded.

`elapsed_time` follows upstream benchmark semantics and includes model prefill
plus generation. The benchmark scripts run one complete workload warmup before
the reported interval, so model initialization, ACLGraph capture, and the first
replay check are excluded consistently for every backend. Direct API calls do
not perform this benchmark warmup automatically. Native result metadata also
separates `prefill_elapsed_seconds` and `decode_elapsed_seconds`.

## Native API

```python
from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams


def main():
    config = PEARLConfig(
        draft_model_path="/data/shared-models/Qwen2.5-0.5B-Instruct",
        target_model_path="/data/shared-models/Qwen2.5-14B-Instruct",
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=2,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        prefill_chunk_size=16,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        gamma=3,
    )
    params = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=False)
    with PEARLEngine(config) as engine:
        engine.add_request("What is 2 + 2?", params)
        output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
        print(output_text, num_tokens, num_acc_tokens, elapsed_time)


if __name__ == "__main__":
    main()
```

The `__main__` guard is required by Python's `spawn` multiprocessing mode. The
controller sets an automatic HCCL NPU socket-port range so it can coexist with
other HCCL jobs in the same container. Worker initialization and generation use
a 300-second deadline by default; set `worker_timeout_seconds` in `PEARLConfig`
or `--worker-timeout-seconds` in the example CLI to change it.

`PEARLConfig.draft_config`, `target_config`, `eos`, and `world_size`, plus
`PEARLEngine.log()`, retain the corresponding upstream compatibility surface.

The runnable version is
`examples/offline_inference_nano_pearl.py`. For the validated Qwen2.5 pair:

```bash
cd /root/data/vllm-ascend-hust
ASCEND_RT_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. \
  /root/miniconda3/envs/vllm-hust-dev/bin/python \
  examples/offline_inference_nano_pearl.py \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 3 \
  --max-model-len 1024 --max-num-seqs 32 --max-tokens 64 \
  'What is 2 + 2?'
```

Set `--mode target-ar` for target-only autoregressive generation or
`--mode bench --num-pearl-steps 100` for the upstream fixed-step benchmark.
Set `--gamma -1` to profile and select gamma automatically. Positive
`--temperature` values use target sampling; the draft remains greedy, matching
the current upstream implementation.

## Direct Worker CLI

The lower-level runtime can still be launched under `torchrun`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  -m vllm_ascend.spec_decode.pearl.native_engine \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 3 \
  --max-model-len 1024 --max-tokens 64 \
  --prompt 'What is 2 + 2?'
```

The direct CLI also accepts GSM8K parquet input through `--gsm8k`, and supports
`--temperature`, `--ignore-eos`, `--seed`, prefix-cache control, ACLGraph
control, `--gpu-memory-utilization`, an explicit `--num-kvcache-blocks`,
batched PEARL, and target AR mode.

The JSON summary reports both raw draft-token acceptance and upstream MAT.
`aggregate_acceptance_rate` is accepted draft tokens divided by verified draft
tokens. `aggregate_mat` is the mean of each request's `num_acc_tokens` segments,
matching upstream benchmark scripts; these metrics are not interchangeable.
Per-request metadata also reports `aclgraph_captures`, `aclgraph_replays`, and
`aclgraph_failed_captures`, plus `aclgraph_capture_attempts` and
`aclgraph_capacity_fallbacks`, plus `aclgraph_shape_fallbacks`. A failed
first-replay check disables that graph shape and falls back to eager execution
instead of returning unchecked tokens. The controller aggregates these
counters across every draft and target worker.

For comparisons against production vLLM-Ascend, use the benchmark entry points
below. They apply the same chat template, warm the complete measured request
set, and emit an output-token SHA-256 digest. `--output-json` persists the same
final payload before it is printed. The serial entry point runs draft TP1 and
target TP1 in the same vLLM process on one NPU; it is not the disjoint PEARL
topology. The example below uses the first 50 GSM8K prompts so batch 32 runs as
one full 32-request chunk followed by an 18-request chunk.

```bash
OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=0 \
  python examples/benchmark_nano_pearl_serial_speculative.py \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 1 --gamma 4 \
  --max-model-len 1024 --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --num-prompts 50 --max-tokens 512 \
  --gsm8k /data/datasets/gsm8k/test.parquet \
  --output-json /tmp/nano-pearl-serial-b32.json

OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2 \
  python examples/benchmark_nano_pearl_target_only.py \
  --model /data/shared-models/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 --max-model-len 1024 --max-num-seqs 32 \
  --batch-sizes 32 --num-prompts 50 --max-tokens 512 \
  --gsm8k /data/datasets/gsm8k/test.parquet \
  --output-json /tmp/nano-pearl-target-b32.json

OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2 \
  HCCL_NPU_SOCKET_PORT_RANGE=auto \
  python examples/benchmark_nano_pearl_speculative.py \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 3 \
  --max-model-len 1024 --batch-sizes 32 \
  --num-prompts 50 --max-tokens 512 \
  --max-aclgraph-entries 64 --pad-finished-requests \
  --draft-use-paged-attention --target-verification-graph-buckets 6 \
  --gsm8k /data/datasets/gsm8k/test.parquet \
  --output-json /tmp/nano-pearl-b32.json
```

For batch 16, use `--batch-sizes 16 --num-prompts 30` and set
`--max-num-seqs 16` for the production and serial baselines. An ACLGraph entry
limit of 32 covers the observed PEARL batch-16 shapes; batch 32 uses 64 so the
32-request and 18-request chunks cannot exhaust the capture budget.

Production vLLM-Ascend cannot evenly partition Qwen2.5-14B's 40 Q heads and
8 KV heads at TP3. Native PEARL applies the upstream dynamic-TP layout instead:
it pads the global layout to 45 Q and 9 KV heads. Target ranks 0 and 1 each
load 15 Q plus 3 KV heads; rank 2 loads 10 valid plus 5 zero-padded Q heads and
2 valid plus 1 zero-padded KV head. Its corresponding attention-out columns
are padded with zeros as well. An experiment that physically removed rank 2's
zero heads changed greedy output and acceptance behavior for only a sub-percent
B4 latency difference, so the stable runtime retains the padded shape.

## Ascend Runtime Mapping

Groups are created in a globally fixed order:

```text
draft group:          [0, ..., draft_tp - 1]
target group:         [draft_tp, ..., world_size - 1]
verification group:   [draft leader, all target ranks]
```

Packed prefill and target-only autoregressive decode use CANN paged attention.
Layers with the same PA operator shape share one graph workspace instead of
retaining one scratch allocation per transformer layer.
PEARL packed target verification uses the TND fused-infer attention path
selected by vLLM-Ascend for speculative decoding. Draft decode uses the same
path by default and can optionally use paged attention for its one-token
steps. For `gamma <= 16`, these speculative calls run in
`torch.npu.NPUGraph`; their request-level query boundaries, KV lengths, page
tables, and shared operator workspaces are rebound before replay. Larger gamma
values exceed the CANN TND speculative-query limit and use the correct eager
paged-attention fallback. Qwen3 also shares the production QKV/QK-Norm/RoPE
fusion when its model shape supports that operator. Dense SDPA exists only as
the CPU test fallback.

## SpecRhythm SLO Planner and Native Dual-Batch Runtime

The vLLM V1 path now exposes the first SpecRhythm control-plane layer. Enable it
with `spec_rhythm=true` in `speculative_config` and provide a request SLO through
`SamplingParams.extra_args`:

```bash
--speculative-config '{
  "method": "draft_model",
  "model": "/data/shared-models/Qwen2.5-0.5B-Instruct",
  "num_speculative_tokens": 5,
  "draft_tensor_parallel_size": 1,
  "spec_rhythm": true,
  "spec_rhythm_min_gamma": 1,
  "spec_rhythm_max_gamma": 5,
  "spec_rhythm_max_eager_tokens": 2,
  "spec_rhythm_roofline": {"8:1": 3, "default": 5}
}'
```

The generic vLLM V1 scheduler integration emits the budget and two-home-batch
metadata for schedulers that own the model workers. It remains metadata-only in
that generic path because vLLM V1 cannot create one cross-engine HCCL group.
The native Ascend PEARL engine is the executable implementation: pass
`enable_spec_rhythm=True` together with continuous batching and preemptive
scheduling. It keeps two persistent logical batches, runs target verification
and the opposite draft window in the same service step, and exchanges a
self-describing HCCL envelope containing proposal id, request id, epoch, width,
and quantized confidence. A rejected parent invalidates the entire eager
continuation; only a fully accepted parent with a matching prefix epoch is
promoted. Variable per-request draft budgets are padded to the configured
gamma only at the device envelope boundary, so target graph shapes remain
bounded. Per-request options use the native `SamplingParams` fields shown below
or the generic nested form `{"spec_rhythm": {"slo_tpot_ms": 50,
"slo_class": "tight"}}`.

The native HCCL PEARL controller exposes the same SLO fields directly on its
per-request `SamplingParams`. Enable SLO-aware admission only with continuous
batching and preemptive scheduling:

```python
engine = PEARLEngine(PEARLConfig(
    draft_model_path="/data/shared-models/Qwen2.5-0.5B-Instruct",
    target_model_path="/data/shared-models/Qwen2.5-14B-Instruct",
    draft_tensor_parallel_size=1,
    target_tensor_parallel_size=3,
    max_num_seqs=64,
    max_num_queued_seqs=256,
    enable_continuous_batching=True,
    enable_preemptive_scheduling=True,
    enable_spec_rhythm=True,
))
engine.add_request(
    prompt,
    SamplingParams(max_tokens=512, temperature=0.0, slo_tpot_ms=50.0,
                   slo_class="tight"),
)
```

Native results include `observed_tpot_ms`, `slo_attained`, and
`slo_goodput_tokens`. They are measured over the decode interval (prefill is
reported separately), matching the paper's decode-goodput definition. The
runtime supports rolling arrivals through `arrival_ts`, request-level gamma
caps, draft-confidence shaping, and synchronized profiling of draft compute,
both HCCL directions, target verification, and wait/state-update time. The
target graph uses a bounded padded bucket while the logical verification width
remains per request.

## OpenAI-Compatible Bridge

The bridge is registered through vLLM's
`speculative_config.method="custom_class"` interface. Its engines cannot form
one cross-engine HCCL group, so it transports prompt and token IDs over a
mode-`0600` Unix socket while the target keeps vLLM-Ascend TP, paged attention,
verification, and ACLGraph:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python \
  -m vllm_ascend.spec_decode.pearl.launcher \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --draft-devices 0,1 --draft-tensor-parallel-size 2 \
  --target-devices 2,3 --num-speculative-tokens 4 \
  --draft-llm-kwargs '{"dtype":"bfloat16","gpu_memory_utilization":0.70}' \
  -- \
  /data/shared-models/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16 --port 8000
```

The bridge is greedy because it transfers draft token IDs rather than draft
probability distributions. Use the native API for upstream PEARL stochastic
target verification.

## Upstream TODO Boundary

The reference repository at commit `6b1cebf` leaves the following items as
project TODOs; they are outside the paper's released runtime and are not
required for nano-PEARL/SpecRhythm inference parity:

- PEARL-2 draft-model training or distillation;
- upstream CUDA-specific packaging and benchmark automation.

The Ascend native runtime implements the inference-side items that the
reference TODO list leaves open: finite-queue continuous batching, chunked
prefill, online acceptance/confidence-based gamma shaping, and request SLO
admission. These are Ascend backend extensions, so they are exposed through
`NativePearlConfig` and do not change the upstream CUDA API. The generic vLLM
V1 scheduler remains metadata-only because it cannot create a cross-engine
HCCL group; executable two-model scheduling is provided by the native PEARL
entrypoint documented above.

The native runtime, like upstream, is text-only and does not add vLLM features
such as multimodal input, LoRA routing, or structured-output constraints.

## Verification

CPU migration tests:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q \
  tests/ut/spec_decode/test_pearl.py \
  tests/ut/spec_decode/test_pearl_bridge.py \
  tests/ut/spec_decode/test_pearl_native.py \
  tests/ut/spec_decode/test_pearl_vocab.py
```

HCCL protocol smoke test:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m vllm_ascend.spec_decode.pearl.smoke
```

Validated NPU paths include Qwen2.5 heterogeneous-vocabulary TP1+TP2, Qwen3
greedy and positive-temperature PEARL, Qwen3 target AR, fixed-step batched
benchmark generation, automatic gamma, Qwen3 dynamic TP3+TP3, and Llama PEARL
weight loading and generation. The speculative FIA ACLGraph path has an
identical-model oracle at batch sizes 1, 8, and 32: Qwen3-0.6B reaches at least
99.62% raw acceptance with zero failed graph captures. The production
QKV/QK-Norm/RoPE fusion was additionally validated on Qwen3-0.6B TP1 plus
Qwen3-32B TP2 with bit-identical output hashes across repeated B32 runs. With
the default AIV HCCL mode, those two B32 runs averaged 1015.33 worker token/s
and 994.45 controller end-to-end token/s.

For Qwen2 models, native PEARL defaults to the production
`vllm::npu_rotary_embedding` operator on both model groups. The draft and
target switches are independent and automatically fall back when the operator
is unavailable. Biased QKV weights intentionally remain in ND: a draft
FRACTAL_NZ experiment improved isolated graph replay latency but changed the
full-run token hash and acceptance behavior, so it is not a runtime mode.

On one three-NPU test host, the Qwen2.5-0.5B TP1 draft plus Qwen2.5-14B TP2
target generated 512 tokens for the leading GSM8K test prompts with gamma 4:

| Active batch | Prompts | vLLM-Ascend target-only e2e | Native PEARL e2e | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| 64 | 100 | 1683.03 token/s | 1738.97 token/s | +3.32% |
| 128 | 200 | 2383.82 token/s | 2402.28 token/s | +0.77% |

Both native runs used a finite prefilled queue, full/half active batch buckets,
32 target-verification graph buckets, and 32 graph entries. Performance is
hardware and prompt dependent; keep the target-only warmup and measurement
procedure identical when reproducing these comparisons.
