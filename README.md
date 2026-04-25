# Dummy Inference Runtimes

- `vllm-dummy/vllm_dummy.py` is a runnable toy that shows disaggregated prefill/decode, a paged KV cache, and paged attention over cached tokens.
- `sglang-dummy/sglang_dummy.py` is a runnable toy that shows radix-tree prefix matching and how prefix reuse reduces prefill work.

Run either demo with `python3 <path-to-script>`.
