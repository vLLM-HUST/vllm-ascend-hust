# DeltaKV source recovery, 2026-10-19

The former parent gitlink `a66dc5310149dda2240ea0f58944a3a5c3c0e4a9`
cannot be fetched from vLLM-HUST/vllm-ascend-hust (Git: not our ref;
GitHub commit endpoint: HTTP 422). Its integration branch no longer exists.

This branch deliberately restores runtime source from upstream v0.23.0 tag commit
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`, matching the official carrier
version. It does not claim to reproduce the missing commit. No operator or
kernel patches beyond selecting this released source are introduced.

The recovery commit is based on the fork's main commit `212a949` and preserves
its `.github/workflows` tree unchanged. CI is not part of this runtime recovery;
the credential does not grant workflow editing. All other files are the tag
tree, plus this provenance note. This is a source-version reset on the
project feature branch only, not a change to main.

The required attention connector load/save hooks exist in this source.
The parent repository must requalify imports, graph-mode serving, payload
installation and generated-token correctness before admitting timing results.
Static hook presence alone is not runtime or performance evidence.
