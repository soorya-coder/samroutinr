#!/usr/bin/env python3
"""
sustain.py - Verify a website can sustain a fixed target TPS (default: 200).

Different question from tps.py. tps.py asks "how fast can this go?" (closed
model - workers loop as fast as they can). This asks "can it hold 200 req/s?"
(open model - requests arrive on a fixed schedule regardless of how slow the
server gets), then gives a PASS/FAIL verdict.

Why the open model matters
--------------------------
If you cap a closed-loop test at 200 rps and the server slows down, your
workers block on slow responses and the load you actually apply silently drops
below 200. You then report "200 tps, all green" for a test that never ran at
200. That is coordinated omission. Here, request i is scheduled for
t0 + i/target no matter what, and the gap between when it *should* have started
and when it did start is measured as "schedule lag" - the honest signal that
either the server or the load generator has saturated.

Sizing
------
Little's Law: concurrency = arrival_rate x latency. 200 tps at 50 ms needs
~10 workers in flight; at 500 ms it needs ~100. --auto (default) runs a short
probe to measure latency, then sizes the pool with headroom. Wrong-sized pools
are the #1 way fixed-rate tests lie, so the script refuses to silently under
provision - it reports INCONCLUSIVE instead of FAIL when it ran out of workers.

Examples
--------
  # the headline case: can example.com hold 200 tps for a minute?
  python sustain.py https://example.com

  # 500 tps for 5 minutes, ignore the first 30s while caches/JIT warm up
  python sustain.py https://api.example.com -T 500 -d 300 --warmup 30

  # enforce a latency SLO and an error budget, for CI
  python sustain.py https://api.example.com/health -T 200 --slo-p95 250 --max-error-rate 0.005

  # POST, fixed worker count, JSON out
  python sustain.py https://api.example.com/orders -T 200 -X POST \
      -H "Content-Type: application/json" -b '{"sku":"A1"}' -w 64 --json
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass

from tps import STOP, Budget, Stats, Target, human_bytes, pct
from tps import worker as closed_loop_worker


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Sample:
    """One transaction. `t_done` lets us slice out a steady-state window later."""

    t_done: float       # perf_counter when the response completed
    latency: float      # seconds, request -> body drained
    lag: float          # seconds late vs. its scheduled arrival time
    status: int | None
    nbytes: int
    err: str | None


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: list[Sample] = []

    def add(self, s: Sample) -> None:
        with self.lock:
            self.samples.append(s)

    def snapshot(self) -> list[Sample]:
        with self.lock:
            return list(self.samples)


# --------------------------------------------------------------------------- #
# open-model load generation
# --------------------------------------------------------------------------- #
class Schedule:
    """
    Hands out fixed arrival slots: request i is due at t0 + i/rate.

    Workers pull the next slot, wait for it, then fire. Because slots are
    precomputed rather than derived from when the previous request finished,
    server slowdown shows up as lag instead of as reduced offered load.
    """

    def __init__(self, t0: float, rate: float, limit: int):
        self.t0 = t0
        self.interval = 1.0 / rate
        self.limit = limit
        self.seq = 0
        self.lock = threading.Lock()

    def next_slot(self) -> float | None:
        with self.lock:
            if self.seq >= self.limit:
                return None
            slot = self.t0 + self.seq * self.interval
            self.seq += 1
            return slot

    @property
    def issued(self) -> int:
        with self.lock:
            return self.seq


def open_loop_worker(target: Target, sched: Schedule, rec: Recorder) -> None:
    conn: http.client.HTTPConnection | None = None
    while not STOP.is_set():
        slot = sched.next_slot()
        if slot is None:
            break

        # Wait for this request's scheduled arrival. If we're already past it,
        # fire immediately and bank the overshoot as lag.
        now = time.perf_counter()
        if slot > now:
            if STOP.wait(slot - now):
                break
            lag = 0.0
        else:
            lag = now - slot

        t0 = time.perf_counter()
        status: int | None = None
        nbytes = 0
        err: str | None = None
        try:
            if conn is None:
                conn = target.connect()
            conn.request(target.method, target.path, body=target.body, headers=target.headers)
            resp = conn.getresponse()
            status = resp.status
            nbytes = len(resp.read())  # drain so the connection can be reused
            if resp.will_close:
                conn.close()
                conn = None
        except Exception as exc:  # noqa: BLE001 - any failure is a failed txn
            err = type(exc).__name__
            detail = str(exc).strip()
            if detail:
                err = f"{err}: {detail[:60]}"
            if conn is not None:
                try:
                    conn.close()
                finally:
                    conn = None

        done = time.perf_counter()
        rec.add(Sample(done, done - t0, lag, status, nbytes, err))

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# worker sizing
# --------------------------------------------------------------------------- #
def probe_latency(target: Target, n: int, workers: int) -> tuple[float, float, int]:
    """Short closed-loop burst to estimate latency. Returns (p50, p95, failures)."""
    STOP.clear()
    stats = Stats()
    budget = Budget(n)
    stats.started = time.perf_counter()
    threads = [threading.Thread(target=closed_loop_worker,
                                args=(i, target, stats, budget, None, 0.0), daemon=True)
               for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(target.timeout * 4 + 5)
    stats.finished = time.perf_counter()

    # Each worker's first request pays DNS + TCP + TLS, which is not
    # representative of steady-state latency on a keep-alive connection.
    # Leaving those in inflates p95 and massively over-sizes the pool.
    raw = stats.latencies
    steady = raw[workers:] if len(raw) > workers * 2 else raw
    lat = sorted(steady)
    if not lat:
        return 0.0, 0.0, n
    return pct(lat, 50), pct(lat, 95), stats.failed


def sizing_latency(p50: float, p95: float) -> float:
    """
    Robust latency estimate for Little's Law.

    Sizing on p95 alone is hostage to a single outlier (one 600 ms hiccup in a
    30-request probe asks for 10x the threads actually needed). Sizing on p50
    alone under-provisions when the tail is genuinely fat. Clamping p95 to 3x
    p50 keeps real tail latency in play while capping outlier blast radius.
    """
    return max(p50, min(p95, p50 * 3.0))


def size_pool(target_rps: float, latency_s: float, headroom: float,
              cap: int) -> tuple[int, int]:
    """Little's Law + headroom. Returns (chosen, ideal_uncapped)."""
    ideal = max(2, math.ceil(target_rps * max(latency_s, 0.001) * headroom))
    return min(ideal, cap), ideal


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(samples: list[Sample], window: tuple[float, float],
              args: argparse.Namespace, workers: int) -> dict:
    w_start, w_end = window
    inwin = [s for s in samples if w_start <= s.t_done <= w_end]
    wall = max(w_end - w_start, 1e-9)

    lat = sorted(s.latency for s in inwin)
    lags = sorted(s.lag for s in inwin)
    ok = [s for s in inwin if s.err is None and s.status is not None and s.status < 400]
    errs: dict[str, int] = {}
    codes: dict[str, int] = {}
    for s in inwin:
        if s.err is not None:
            errs[s.err] = errs.get(s.err, 0) + 1
        else:
            codes[str(s.status)] = codes.get(str(s.status), 0) + 1

    n = len(inwin)
    achieved = n / wall
    err_rate = (n - len(ok)) / n if n else 1.0

    return {
        "url": args.url,
        "method": args.method.upper(),
        "target_tps": args.target,
        "workers": workers,
        "measured_window_s": round(wall, 3),
        "warmup_s": args.warmup,
        "requests": n,
        "successful": len(ok),
        "failed": n - len(ok),
        "achieved_tps": round(achieved, 2),
        "achieved_pct_of_target": round(100.0 * achieved / args.target, 1),
        "successful_tps": round(len(ok) / wall, 2),
        "error_rate": round(err_rate, 5),
        "throughput_bytes_per_s": round(sum(s.nbytes for s in inwin) / wall, 1),
        "latency_ms": {
            "min": round(lat[0] * 1000, 2) if lat else 0.0,
            "avg": round(statistics.fmean(lat) * 1000, 2) if lat else 0.0,
            "p50": round(pct(lat, 50) * 1000, 2),
            "p90": round(pct(lat, 90) * 1000, 2),
            "p95": round(pct(lat, 95) * 1000, 2),
            "p99": round(pct(lat, 99) * 1000, 2),
            "max": round(lat[-1] * 1000, 2) if lat else 0.0,
        },
        "schedule_lag_ms": {
            "p50": round(pct(lags, 50) * 1000, 2),
            "p95": round(pct(lags, 95) * 1000, 2),
            "max": round(lags[-1] * 1000, 2) if lags else 0.0,
        },
        "status_codes": dict(sorted(codes.items())),
        "errors": dict(sorted(errs.items(), key=lambda kv: -kv[1])),
    }


def verdict(s: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    """Returns (PASS | FAIL | INCONCLUSIVE, reasons)."""
    reasons: list[str] = []
    hit_rate = s["achieved_tps"] >= args.target * args.tolerance
    lag_p95_s = s["schedule_lag_ms"]["p95"] / 1000.0

    # Was the generator itself the bottleneck? Little's Law on observed latency
    # says how many workers this rate needed; if we had fewer, the test under
    # applied load and a FAIL verdict would be unearned.
    needed = math.ceil(args.target * max(s["latency_ms"]["p50"] / 1000.0, 0.001))
    starved = needed > s["workers"] and lag_p95_s > args.max_lag

    if not hit_rate and starved:
        reasons.append(
            f"load generator saturated: {s['workers']} workers, but {args.target:g} tps "
            f"at p50 {s['latency_ms']['p50']:.0f} ms needs ~{needed}. "
            f"Re-run with -w {needed * 2} (or --headroom higher)."
        )
        return "INCONCLUSIVE", reasons

    if not hit_rate:
        reasons.append(f"only sustained {s['achieved_tps']:.1f} tps of "
                       f"{args.target:g} target ({s['achieved_pct_of_target']:.0f}%)")
    if s["error_rate"] > args.max_error_rate:
        reasons.append(f"error rate {s['error_rate'] * 100:.2f}% exceeds "
                       f"{args.max_error_rate * 100:.2f}% budget")
    if args.slo_p95 and s["latency_ms"]["p95"] > args.slo_p95:
        reasons.append(f"p95 latency {s['latency_ms']['p95']:.0f} ms exceeds "
                       f"SLO of {args.slo_p95:.0f} ms")
    if lag_p95_s > args.max_lag and hit_rate:
        reasons.append(f"schedule lag p95 {s['schedule_lag_ms']['p95']:.0f} ms - "
                       f"requests are queueing, target is near the ceiling")

    if not s["requests"]:
        return "INCONCLUSIVE", ["no requests completed inside the measurement window"]
    return ("PASS" if not reasons else "FAIL"), reasons


def print_report(s: dict, status: str, reasons: list[str]) -> None:
    L, G = s["latency_ms"], s["schedule_lag_ms"]
    bar = "=" * 64
    print()
    print(bar)
    print(f"  {status}  -  {s['method']} {s['url']}")
    print(bar)
    print(f"  Target             : {s['target_tps']:g} tps")
    print(f"  Achieved           : {s['achieved_tps']:.2f} tps "
          f"({s['achieved_pct_of_target']:.0f}% of target)")
    print(f"  Successful         : {s['successful_tps']:.2f} tps")
    print(f"  Workers            : {s['workers']}")
    print(f"  Measured window    : {s['measured_window_s']:.1f} s"
          + (f"  (after {s['warmup_s']:g}s warm-up)" if s["warmup_s"] else ""))
    print(f"  Requests           : {s['requests']}  "
          f"({s['successful']} ok / {s['failed']} failed, "
          f"{s['error_rate'] * 100:.2f}% errors)")
    print(f"  Throughput         : {human_bytes(s['throughput_bytes_per_s'])}/s")
    print()
    print("  Latency (ms)")
    print(f"    min {L['min']:>8.2f}   avg {L['avg']:>8.2f}   max {L['max']:>8.2f}")
    print(f"    p50 {L['p50']:>8.2f}   p90 {L['p90']:>8.2f}   "
          f"p95 {L['p95']:>8.2f}   p99 {L['p99']:>8.2f}")
    print()
    print("  Schedule lag (ms) - how late requests were vs. the fixed schedule")
    print(f"    p50 {G['p50']:>8.2f}   p95 {G['p95']:>8.2f}   max {G['max']:>8.2f}")
    if s["status_codes"]:
        print()
        print("  Status codes")
        for code, count in s["status_codes"].items():
            print(f"    {code:<6} {count}")
    if s["errors"]:
        print()
        print("  Errors")
        for name, count in s["errors"].items():
            print(f"    {count:>6}  {name}")
    print()
    if reasons:
        print(f"  Why {status}:")
        for r in reasons:
            print(f"    - {r}")
    else:
        print(f"  Sustained {s['target_tps']:g} tps within all thresholds.")
    print(bar)


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def progress(rec: Recorder, sched: Schedule, t0: float, t_end: float,
             interval: float) -> None:
    last = 0
    last_t = t0
    rate = 1.0 / sched.interval
    while not STOP.wait(interval):
        now = time.perf_counter()
        samples = rec.snapshot()
        n = len(samples)
        recent = samples[last:n]
        inst = (n - last) / max(now - last_t, 1e-9)
        avg_ms = (statistics.fmean(s.latency for s in recent) * 1000) if recent else 0.0
        lag_ms = (max(s.lag for s in recent) * 1000) if recent else 0.0
        # Completions owed vs. the schedule. Compare against slots that are
        # actually *due* by now - sched.issued counts slots workers have
        # claimed but are still sleeping on, which is always ~= worker count.
        due = min(sched.limit, int((now - t0) * rate))
        sys.stderr.write(f"\r  [{now - t0:6.1f}s] {inst:7.1f} tps  "
                         f"avg {avg_ms:7.1f} ms  lag {lag_ms:7.1f} ms  "
                         f"behind {max(0, due - n):5d}  "
                         f"{max(0.0, t_end - now):5.1f}s left   ")
        sys.stderr.flush()
        last, last_t = n, now


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sustain.py",
        description="Verify a website can sustain a fixed target TPS (open model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Only load-test systems you own or are authorized to test.",
    )
    p.add_argument("url", help="target URL")
    p.add_argument("-T", "--target", type=float, default=200.0,
                   help="target requests/sec to sustain (default: 200)")
    p.add_argument("-d", "--duration", type=float, default=60.0,
                   help="seconds to hold the target, warm-up included (default: 60)")
    p.add_argument("--warmup", type=float, default=5.0, metavar="SEC",
                   help="exclude the first SEC from the verdict (default: 5)")
    p.add_argument("-w", "--workers", type=int,
                   help="fixed worker count; default is auto-sized from a probe")
    p.add_argument("--max-workers", type=int, default=512,
                   help="ceiling for auto-sizing (default: 512)")
    p.add_argument("--headroom", type=float, default=2.0,
                   help="multiply the Little's Law worker estimate by this (default: 2)")
    p.add_argument("--probe", type=int, default=30, metavar="N",
                   help="requests in the sizing probe, 0 to skip (default: 30)")

    p.add_argument("-X", "--method", default="GET", help="HTTP method (default: GET)")
    p.add_argument("-H", "--header", action="append", default=[], metavar="'K: V'",
                   help="extra request header; repeatable")
    p.add_argument("-b", "--body", help="request body string")
    p.add_argument("--body-file", help="read request body from a file")
    p.add_argument("-t", "--timeout", type=float, default=10.0,
                   help="per-request timeout in seconds (default: 10)")
    p.add_argument("-k", "--insecure", action="store_true",
                   help="skip TLS certificate verification")

    p.add_argument("--tolerance", type=float, default=0.98, metavar="FRAC",
                   help="fraction of target that still counts as met (default: 0.98)")
    p.add_argument("--max-error-rate", type=float, default=0.01, metavar="FRAC",
                   help="error budget as a fraction (default: 0.01 = 1%%)")
    p.add_argument("--slo-p95", type=float, metavar="MS",
                   help="fail if p95 latency exceeds this many ms")
    p.add_argument("--max-lag", type=float, default=0.5, metavar="SEC",
                   help="schedule-lag p95 above this means saturation (default: 0.5)")

    p.add_argument("--interval", type=float, default=1.0,
                   help="live progress refresh, 0 to disable (default: 1)")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON only")

    args = p.parse_args(argv)
    if "://" not in args.url:
        args.url = "https://" + args.url
    if args.target <= 0:
        p.error("--target must be > 0")
    if args.duration <= 0:
        p.error("--duration must be > 0")
    if args.warmup < 0 or args.warmup >= args.duration:
        p.error("--warmup must be >= 0 and less than --duration")
    if args.workers is not None and args.workers < 1:
        p.error("--workers must be >= 1")
    if args.body and args.body_file:
        p.error("use either --body or --body-file, not both")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = (lambda *a: None) if args.json else \
        (lambda *a: print(*a, file=sys.stderr, flush=True))

    headers: dict[str, str] = {}
    for h in args.header:
        if ":" not in h:
            print(f"error: bad header {h!r}, expected 'Name: value'", file=sys.stderr)
            return 2
        name, _, value = h.partition(":")
        headers[name.strip()] = value.strip()

    body: bytes | None = None
    if args.body_file:
        with open(args.body_file, "rb") as fh:
            body = fh.read()
    elif args.body:
        body = args.body.encode()

    try:
        target = Target(args.url, args.method, headers, body, args.timeout, args.insecure)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ---- phase 1: size the worker pool ---------------------------------- #
    workers = args.workers
    if workers is None:
        if args.probe > 0:
            log(f"Probing {args.url} ({args.probe} requests) to size the pool...")
            p50, p95, fails = probe_latency(target, args.probe,
                                            min(10, args.max_workers))
            if p95 <= 0:
                print("error: probe got no successful responses; is the URL reachable?",
                      file=sys.stderr)
                return 2
            lat_est = sizing_latency(p50, p95)
            workers, ideal = size_pool(args.target, lat_est, args.headroom,
                                       args.max_workers)
            log(f"  probe: p50 {p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms"
                + (f", {fails} failed" if fails else ""))
            log(f"  sizing: {args.target:g} tps x {lat_est * 1000:.0f} ms "
                f"x {args.headroom:g} headroom -> {ideal} workers"
                + (f", capped to {workers}" if workers < ideal else ""))
            if workers < ideal:
                log(f"  WARNING: --max-workers {args.max_workers} may under-apply load")
        else:
            workers = min(args.max_workers, max(1, math.ceil(args.target / 4)))
            log(f"Probe skipped; using {workers} workers")

    total = max(1, int(round(args.target * args.duration)))
    log(f"Sustaining {args.target:g} tps for {args.duration:g}s "
        f"({total} requests, {workers} workers, "
        f"{args.warmup:g}s warm-up excluded from the verdict)")

    # ---- phase 2: hold the target rate ---------------------------------- #
    STOP.clear()
    rec = Recorder()
    t0 = time.perf_counter()
    t_end = t0 + args.duration
    sched = Schedule(t0, args.target, total)

    threads = [threading.Thread(target=open_loop_worker, args=(target, sched, rec),
                                daemon=True, name=f"w{i}") for i in range(workers)]
    reporter = None
    if args.interval > 0 and not args.json:
        reporter = threading.Thread(target=progress,
                                    args=(rec, sched, t0, t_end, args.interval),
                                    daemon=True)
    for t in threads:
        t.start()
    if reporter:
        reporter.start()

    interrupted = False
    try:
        # Hold until the schedule is exhausted or the clock runs out, then let
        # in-flight requests drain so their latency isn't truncated.
        while time.perf_counter() < t_end and any(t.is_alive() for t in threads):
            STOP.wait(0.05)
    except KeyboardInterrupt:
        interrupted = True
    STOP.set()
    for t in threads:
        t.join(args.timeout + 1)
    t_stop = time.perf_counter()
    if reporter:
        reporter.join(0.5)
        sys.stderr.write("\r" + " " * 88 + "\r")
        sys.stderr.flush()
    if interrupted:
        log("interrupted - reporting the partial window")

    samples = rec.snapshot()
    if not samples:
        print("error: no requests completed", file=sys.stderr)
        return 2

    window = (t0 + args.warmup, min(t_end, t_stop))
    if window[1] <= window[0]:
        window = (t0, t_stop)  # too short to warm up; measure everything
    summary = summarize(samples, window, args, workers)
    status, reasons = verdict(summary, args)
    summary["verdict"] = status
    summary["verdict_reasons"] = reasons

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary, status, reasons)

    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
