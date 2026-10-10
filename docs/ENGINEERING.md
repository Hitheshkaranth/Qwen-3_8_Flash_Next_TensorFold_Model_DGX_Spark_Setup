# Engineering Notes

This is the technical log behind this repository: what was built, what broke, how each problem
was diagnosed, and the raw data behind every number in the README. Everything here was captured
on one **NVIDIA DGX Spark (GB10)**, the same machine that runs the
[Ornith-1.5 deployment](https://github.com/Hitheshkaranth/Ornith-1.5_A3B_Model_DGX_Spark_Setup)
and the [vLLM NVFP4 build of this model](https://github.com/Hitheshkaranth/Qwen3.8-Flash-Next_Model_DGX_Spark_Setup),
on 2026-10-09 and 2026-10-10.

## 1. Hardware profile

```
$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVIDIA GB10, 580.173.02            # container reports CUDA 13.3 forward-compat (driver 610.43.02)

$ free -g | head -2                # TensorFold serving, 8 requests in flight
               total        used        free      shared  buff/cache   available
Mem:             121          84           1           0          37          36

$ nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader
tensorfold-native, 74560 MiB
python3, 1199 MiB                  # the vision helper
```

GPU and CPU share the 121 GiB, so "GPU memory" and "RAM" are the same pool. The 37 GiB of
buff/cache is mostly the checkpoint's page cache and is reclaimable.

## 2. Which checkpoint?

Three ways to fit Qwen3.8-Flash-Next (BF16 335 GiB, FP8 173 GiB) in 121 GiB were on the table:

| Checkpoint | Size | Experts/token | Engine | Outcome |
|---|---|---|---|---|
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | 99 GiB | 10 | vLLM (MiaAI-Lab recipe) | served first: 166.8 tok/s @16, smooth streaming |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 124 GiB | 10 | vLLM | not tried: leaves too little KV at this budget |
| **`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`** | 122 GiB on disk, 65.7 GiB loaded | **5** | **TensorFold** | **this repo**: 400.4 tok/s @16 |

The INT4-AR checkpoint is smaller in memory than on disk: its `ple-table/` (49 GiB, FP8) and
GPTQ shards are repacked at load, and nothing is streamed from SSD at runtime.

The upstream vLLM recipe's README was changed on 2026-10-04 to say "everyone should use the
TensorFold recipe", which prompted this evaluation.

## 3. Local changes to the recipe

Only one line of the recipe is changed, by `download.sh`:

```diff
-      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
+      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-}" \
```

With no host `hf` CLI, `prepare.sh` downloads inside its container and did not forward
`HF_HUB_DISABLE_XET`. Xet transfers stalled on this network (same symptom as every other model on
this box), so plain HTTP is forced. Read timeouts on individual shards still happened during the
122 GiB download; `snapshot_download` resumed them on its own.

`prepare.sh`'s preflight refuses to run with less than `MEM_FLOOR_GIB` available, which is a
serving floor. `download.sh` sets `MEM_CHECK=0` so the download can run while another model is
still serving.

## 4. Serving configuration decisions

- **`PARALLEL=16`** (recipe default 8, maximum 16). Measured: 8 slots → 220.0 tok/s at 8 users;
  16 slots → 272.4 at 8 and **400.4 at 16**; 20 users against 16 slots → 321.6 (queueing).
- **API key on everything.** TensorFold applies `TENSORFOLD_API_KEY` to `/metrics` too (vLLM
  leaves `/metrics` open). The key stays in the environment; `run.sh` never writes it to
  `upstream/.env`.
- **Bind `172.17.0.1` only.** The gateway (host) and Prometheus (Docker) both reach the bridge
  address; the LAN can't.
- **Restart policy.** The recipe's container has none; `run.sh` sets `unless-stopped`, following
  this box's rule that only the live model carries it.
- **The recipe's smoke test** sends no key, so with a key it always reports failure and exits
  non-zero after the server is up. `run.sh` ignores that exit status and warms up with a real,
  authenticated 8K-token request instead.

## 5. Benchmark methodology

Same scripts as the Ornith and vLLM Flash-Next repos, sent directly to `:8006`: the gateway's
per-user cap (4) would otherwise hold a single benchmark client to 4 concurrent requests.

### 5.1 Raw token rate (`benchmarks/throughput_bench.py`)

512 forced output tokens per request (`ignore_eos`, `min_tokens`), temperature 0.6, one warm-up,
then N concurrent requests. MTP acceptance from `tensorfold:mtp_accepted_total /
mtp_drafted_total`: 12,241 / 16,653 = **73.5%**.

### 5.2 Reasoning (`benchmarks/reasoning_bench.py`)

GSM8K first 100, MATH-500 first 100 integer-answer problems, 4 long-form prompts; thinking on,
temperature 0.6 / top_p 0.95 / top_k 20, 16 in flight. The script now retries transient
`datasets-server` 5xx errors, caches datasets in `benchmarks/.cache`, and saves each section as
it finishes (a 502 on MATH-500 once threw away a finished GSM8K run).

**Determinism:** runs at `PARALLEL=8` and `PARALLEL=16` produced identical results to the token:
GSM8K 576.76 avg tokens, MATH-500 1,499.52, long-form `[32768, 16062, 32768, 32768]`. Only wall
times differed. TensorFold's sampling is deterministic per request even at temperature 0.6, so
repeat runs do not average out sampling noise.

### 5.3 Knowledge (`benchmarks/knowledge_bench.py`)

MMLU-Pro, 50 questions per subject × 14 (seed 0), the identical 700 Ornith answered; 8 in flight.
Per-question records go to the `.jsonl`, so the run resumes after an interruption (it was paused
at 193 for a slot-count change). `benchmarks/knowledge_compare.py` scores two runs on their common
questions while one is still running.

### 5.4 Prefill and streaming (`benchmarks/prefill_bench.py`)

Cold prompts start with a random nonce so the prefix cache can't serve them; warm prompts are
resent verbatim; TTFT uses `max_tokens=1`. The mixed test starts N decoding streams, sends one cold
32K prompt, and records every decoder's gap between streamed chunks.

The first version only counted gaps whose *closing* chunk arrived inside the prefill window. It
reported a 54 ms p50 for TensorFold, which looked too good: a stream that freezes for the entire
prefill produces one long gap that closes *after* the window, and was dropped. Counting every gap
that overlaps the window showed the real behaviour: **every running stream paused for the whole
12.8 s prefill** (p95 = max = 12,847 ms, 28 gaps counted). With 8 decoders against 8 slots the
32K prompt waited 144 s for a slot instead; the test uses 7 decoders so it measures prefill, not
queueing.

## 6. Monitoring

TensorFold exports `tensorfold:*` metrics behind the API key. Prometheus reads the key from a
mounted file (`authorization.credentials_file`) and the job rewrites names to `vllm:*` and adds
`model_name`, so the shared vLLM Command Center dashboard needs no TensorFold-specific queries:

- `tensorfold:requests_total{status="200"}` → `vllm:request_success_total` (non-200 dropped)
- everything else `tensorfold:X` → `vllm:X`

One real semantic difference: `generation_tokens_total` only moves when a request **finishes**;
in-flight tokens are in `generation_tokens_running`. During minutes-long reasoning requests the
plain `rate()` read 0 while the engine produced ~170 tok/s. The dashboard's output line and the
**Output Now** / **Peak Output** stat panels use

```
deriv((vllm:generation_tokens_total{job="tensorfold"} + vllm:generation_tokens_running{job="tensorfold"})[1m:5s])
```

(plus plain `rate()` for vLLM jobs). The peak panel is `max_over_time` of that over the
dashboard range; it reads slightly below short benchmark bursts because of the 1-minute window
(376 tok/s over the benchmark period against the benchmark's own 400.4).

If `prometheus.yml` is a single-file bind mount, edit it and `docker restart` the container:
`sed -i` replaces the inode and SIGHUP keeps reading the old one.

## 7. Known limitations / open items

- **Streams pause during prefill.** Any new long prompt stops every other stream until it has
  been read. The vLLM build interleaves prefill chunks instead (~1 s steps).
- **Capability.** 5 of 10 experts: ~10% lower general capability on the checkpoint authors' own
  harness (46.6–47.6 vs 51.8). Not visible on GSM8K/MATH-500 here; MMLU-Pro, finished later, came out at 81.9% vs Ornith 81.1% on the same 700 questions.
- **No structured outputs** (`response_format` → HTTP 400).
- **Recipe smoke test** always "fails" with a key (cosmetic).
- Not yet tried: `PARALLEL` above 16 (the recipe caps it at 16), BF16 KV, `reasoning_effort`
  presets' effect on accuracy and speed.
