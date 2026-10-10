"""Prefill / TTFT benchmark against an OpenAI-compatible vLLM server.

usage: python3 benchmarks/prefill_bench.py <base_url> <model> <tag> [--sizes 8000 32000 64000]
                                           [--decoders 8] [--mixed-context 32000]

1. cold: a unique nonce leads every prompt, so nothing is served from the
   prefix cache; TTFT = queue + prefill (max_tokens=1).
2. warm: the same prompt again, so the prefix cache should serve most of it.
3. mixed: N streams decode (ignore_eos); after 5 s one cold prompt of
   --mixed-context tokens arrives. Reports its TTFT and the decoders'
   inter-chunk gap (p50/p95/max) in the quiet window vs the prefill window.
API key from $GATEWAY_KEY. Appends one JSON line per run to benchmarks/prefill_results.jsonl.
"""
import argparse, json, os, statistics, threading, time, urllib.request, uuid

FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
TASK = "In one short sentence, what kind of document is the log above?"
KEY = os.environ.get("GATEWAY_KEY", "")


def ctx(tokens, nonce):
    return f"Session {nonce}.\n" + "".join(FILLER.format(i=i) for i in range(max(1, tokens // 25)))


def req(base, payload, stream=False, timeout=1800):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(), h)
    return urllib.request.urlopen(r, timeout=timeout)


def ttft(base, model, text):
    t = time.time()
    r = json.load(req(base, {"model": model, "max_tokens": 1,
                             "messages": [{"role": "user", "content": text + "\n" + TASK}]}))
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return time.time() - t, u["prompt_tokens"], cached


def decoder(base, model, gaps, stop):
    p = {"model": model, "stream": True, "max_tokens": 4000, "ignore_eos": True, "temperature": 0,
         "messages": [{"role": "user", "content": "Write a long continuous essay on maritime navigation."}]}
    last = None
    for line in req(base, p, stream=True):
        if stop.is_set():
            break
        if line.startswith(b"data: {"):
            now = time.time()
            if last is not None:
                gaps.append((now, now - last))
            last = now


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 32000, 64000])
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--mixed-context", type=int, default=32000)
    a = ap.parse_args()
    out = {"tag": a.tag, "date": time.strftime("%F %T"), "cold": {}, "warm": {}}

    ttft(a.base, a.model, ctx(2000, uuid.uuid4()))  # warm-up
    for n in a.sizes:
        text = ctx(n, uuid.uuid4())
        s, pt, c = ttft(a.base, a.model, text)
        out["cold"][n] = {"ttft_s": round(s, 2), "prompt_tokens": pt, "cached": c, "tok_s": round(pt / s)}
        print(f"[{a.tag}] cold {pt:6d} tok  TTFT {s:6.2f}s  {pt / s:6.0f} tok/s  cached={c}")
        s, pt, c = ttft(a.base, a.model, text)
        out["warm"][n] = {"ttft_s": round(s, 2), "prompt_tokens": pt, "cached": c}
        print(f"[{a.tag}] warm {pt:6d} tok  TTFT {s:6.2f}s  cached={c}")

    if a.decoders:
        gaps, stop = [], threading.Event()
        ths = [threading.Thread(target=decoder, args=(a.base, a.model, gaps, stop), daemon=True)
               for _ in range(a.decoders)]
        for t in ths:
            t.start()
        time.sleep(8)
        t0 = time.time()
        s, pt, c = ttft(a.base, a.model, ctx(a.mixed_context, uuid.uuid4()))
        t1 = time.time()
        # A gap belongs to a window if the interval (ts - g, ts] overlaps it, so a
        # stream that freezes for the whole prefill still contributes its long gap.
        time.sleep(3)
        stop.set()
        quiet = [g for ts, g in gaps if t0 - 5 <= ts < t0]
        busy = [g for ts, g in gaps if ts > t0 and ts - g < t1]
        out["mixed"] = {"decoders": a.decoders, "prompt_tokens": pt, "ttft_s": round(s, 2),
                        "quiet_gap_ms": [round(1000 * pct(quiet, q)) for q in (0.5, 0.95)],
                        "prefill_gap_ms": [round(1000 * pct(busy, q)) for q in (0.5, 0.95)]
                        + [round(1000 * max(busy))] if busy else None, "prefill_gaps_counted": len(busy)}
        print(f"[{a.tag}] mixed {a.decoders} decoders + cold {pt} tok: TTFT {s:.2f}s | decoder gap "
              f"quiet p50/p95 {out['mixed']['quiet_gap_ms']} ms, during prefill p50/p95/max "
              f"{out['mixed']['prefill_gap_ms']} ms ({len(busy)} gaps)")
    with open(os.path.join(os.path.dirname(__file__), "prefill_results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


main()
