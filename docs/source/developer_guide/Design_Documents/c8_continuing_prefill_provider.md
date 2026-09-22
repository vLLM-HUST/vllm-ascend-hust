# C8 Continuing-Prefill Provider Contract

## Purpose

The C8 attention backend has a native paged INT8 decode path. Cached
multi-token prefill can require a different implementation, because the
built-in compatibility path gathers the selected NZ pages, converts them to a
dense layout, materializes floating-point K/V, and then invokes TND attention.

The provider contract exposes only this narrow execution boundary. It does not
select a quantization policy, alter cache allocation, or activate a provider by
default.

## Activation

Set the optional Ascend additional configuration field to a Python
`module:factory` path:

```json
{
  "c8_continuing_prefill_provider": "my_extension.provider:create_provider"
}
```

The default is `null`. With the default, imports, dispatch, output, and graph
capture follow the existing host path.

The factory is called once per C8 attention layer after weights are loaded. It
receives `C8ContinuingPrefillProviderConfig` and must return an object that
implements `C8ContinuingPrefillProvider`.

## Invocation

The host invokes the provider only after it has identified a cached
multi-token prefill, covering both standalone `PrefillCacheHit` and mixed
`ChunkedPrefill` states. Decode-only and all-new prefill requests remain on
their existing paths. A mixed batch presents all prefill rows to the provider
while the host continues to execute the decode rows with the native paged C8
decode path.

`C8ContinuingPrefillRequest` contains:

- causal multi-token TND query and output views;
- paged five-dimensional NZ INT8 K/V views;
- the prefill block table and valid KV lengths;
- cumulative query lengths;
- TP-local per-channel K/V antiquant scales;
- GQA dimensions, block size, attention scale, mask, sparse mode, and capture
  state.

The cache already contains the new K/V tokens when the provider is invoked.

## Eligibility And Fallback

`is_eligible(request)` is the only decline point. It must be side-effect free:
it cannot mutate request buffers, launch device work, synchronize the device,
or allocate graph workspace. `False` preserves the existing host fallback. In
graph capture it preserves the existing `full_graph_fia` route rather than
partially constructing a different graph.

After returning `True`, `forward(request)` must write the complete prefill
result into the supplied output view and return
`C8ContinuingPrefillResult`. Exceptions, non-boolean eligibility, invalid
results, and non-tensor workspace entries fail closed; the host does not hide
them behind the dense fallback.

## Graph Lifetime

Provider execution runs during ACL graph capture, so replay repeats captured
device work without another Python provider call. The result's `workspace`
tuple declares tensors whose addresses must remain stable for replay. The host
keeps strong references to every capture workspace for the lifetime of the
attention implementation, which matches the cached graph lifetime. Eager
workspace is retained until the next provider invocation.

The provider must not retain request tensors itself. Runtime activation still
requires provider-specific correctness and capture/replay validation against
an accepted host revision.
