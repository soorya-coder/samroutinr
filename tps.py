#!/usr/bin/env python3
"""
tps.py - Measure the TPS (transactions/requests per second) of a website.

Standard library only. Each worker keeps its own persistent HTTP connection
(keep-alive) so you measure the server, not TCP/TLS handshake overhead.

Examples
--------
  # 20 workers hammering a URL for 30 seconds
  python tps.py https://example.com -w 20 -d 30

  # fixed total number of requests instead of a duration
  python tps.py https://example.com -w 50 -n 5000

  # POST with a JSON body and auth header
  python tps.py https://api.example.com/v1/orders -w 10 -d 15 \
      -X POST -H "Authorization: Bearer xyz" -H "Content-Type: application/json" \
      -b '{"sku":"A1","qty":2}'

  # warm up gradually and cap the offered load
  python tps.py https://example.com -w 100 -d 60 --ramp-up 10 --rps 500
"""

from __future__ import annotations

import argparse
import http.client
import json
import ssl
import statistics
import sys
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field

STOP = threading.Event()


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Thread-safe accumulator for per-request outcomes."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    latencies: list[float] = field(default_factory=list)  # seconds
    statuses: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    bytes_in: int = 0
    started: float = 0.0
    finished: float = 0.0

    def record(self, latency: float, status: int | None, nbytes: int, err: str | None) -> None:
        with self.lock:
            self.latencies.append(latency)
            self.bytes_in += nbytes
            if err is not None:
                self.errors[err] += 1
            else:
                self.statuses[status] += 1

    @property
    def total(self) -> int:
        return len(self.latencies)

    @property
    def ok(self) -> int:
        return sum(c for s, c in self.statuses.items() if s is not None and s < 400)

    @property
    def failed(self) -> int:
        return self.total - self.ok


def pct(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[k]


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
class Target:
    """Parsed URL plus everything a worker needs to fire one request."""

    def __init__(self, url: str, method: str, headers: dict[str, str], body: bytes | None,
                 timeout: float, insecure: bool):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme {u.scheme!r} (use http or https)")
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.https else 80)
        self.path = urllib.parse.urlunsplit(("", "", u.path or "/", u.query, ""))
        self.method = method.upper()
        self.body = body
        self.timeout = timeout

        self.headers = {"Host": u.netloc, "User-Agent": "tps.py/1.0",
                        "Accept": "*/*", "Connection": "keep-alive"}
        self.headers.update(headers)
        if body is not None and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))

        if self.https:
            self.ctx = ssl._create_unverified_context() if insecure \
                else ssl.create_default_context()
        else:
            self.ctx = None

    def connect(self) -> http.client.HTTPConnection:
        if self.https:
            return http.client.HTTPSConnection(self.host, self.port,
                                               timeout=self.timeout, context=self.ctx)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)


def worker(wid: int, target: Target, stats: Stats, budget: "Budget",
           limiter: "RateLimiter | None", ramp_delay: float) -> None:
    if ramp_delay and not STOP.wait(ramp_delay):
        pass
    if STOP.is_set():
        return

    conn: http.client.HTTPConnection | None = None
    while not STOP.is_set() and budget.take():
        if limiter is not None:
            limiter.acquire()
            if STOP.is_set():
                break

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
            nbytes = len(resp.read())  # must drain to reuse the connection
            if resp.will_close:
                conn.close()
                conn = None
        except Exception as exc:  # noqa: BLE001 - any failure is just a failed txn
            err = type(exc).__name__
            detail = str(exc).strip()
            if detail:
                err = f"{err}: {detail[:60]}"
            if conn is not None:
                try:
                    conn.close()
                finally:
                    conn = None

        stats.record(time.perf_counter() - t0, status, nbytes, err)

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# load shaping
# --------------------------------------------------------------------------- #
class Budget:
    """Caps total requests. Unlimited when `limit` is None (duration mode)."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.remaining = limit or 0
        self.lock = threading.Lock()

    def take(self) -> bool:
        if self.limit is None:
            return True
        with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True

    @property
    def issued(self) -> int:
        return 0 if self.limit is None else self.limit - self.remaining


class RateLimiter:
    """Simple shared token-bucket to cap offered requests/sec."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_slot = time.perf_counter()

    def acquire(self) -> None:
        with self.lock:
            now = time.perf_counter()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        wait = slot - time.perf_counter()
        if wait > 0:
            STOP.wait(wait)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def progress(stats: Stats, deadline: float | None, budget: Budget, interval: float) -> None:
    last_n, last_t = 0, time.perf_counter()
    while not STOP.wait(interval):
        now = time.perf_counter()
        with stats.lock:
            n = len(stats.latencies)
            recent = stats.latencies[last_n:n]
        inst = (n - last_n) / (now - last_t) if now > last_t else 0.0
        avg_ms = (sum(recent) / len(recent) * 1000) if recent else 0.0
        elapsed = now - stats.started
        if deadline is not None:
            left = f"{max(0.0, deadline - now):5.1f}s left"
        else:
            left = f"{budget.issued}/{budget.limit} sent"
        sys.stderr.write(f"\r  [{elapsed:6.1f}s] {inst:8.1f} tps  "
                         f"avg {avg_ms:7.1f} ms  {n} done  {left}   ")
        sys.stderr.flush()
        last_n, last_t = n, now


def report(stats: Stats, args: argparse.Namespace, latencies: list[float]) -> dict:
    wall = stats.finished - stats.started
    lat = sorted(latencies)
    tps = stats.total / wall if wall > 0 else 0.0
    ok_tps = stats.ok / wall if wall > 0 else 0.0

    summary = {
        "url": args.url,
        "method": args.method.upper(),
        "workers": args.workers,
        "duration_s": round(wall, 3),
        "requests": stats.total,
        "successful": stats.ok,
        "failed": stats.failed,
        "tps": round(tps, 2),
        "tps_successful": round(ok_tps, 2),
        "throughput_bytes_per_s": round(stats.bytes_in / wall, 1) if wall > 0 else 0.0,
        "latency_ms": {
            "min": round(lat[0] * 1000, 2) if lat else 0.0,
            "avg": round(statistics.fmean(lat) * 1000, 2) if lat else 0.0,
            "p50": round(pct(lat, 50) * 1000, 2),
            "p90": round(pct(lat, 90) * 1000, 2),
            "p95": round(pct(lat, 95) * 1000, 2),
            "p99": round(pct(lat, 99) * 1000, 2),
            "max": round(lat[-1] * 1000, 2) if lat else 0.0,
        },
        "status_codes": {str(k): v for k, v in sorted(stats.statuses.items(),
                                                     key=lambda kv: kv[0] or 0)},
        "errors": dict(stats.errors.most_common()),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return summary

    L = summary["latency_ms"]
    print()
    print("=" * 62)
    print(f"  {summary['method']} {summary['url']}")
    print("=" * 62)
    print(f"  Workers            : {summary['workers']}")
    print(f"  Wall time          : {summary['duration_s']:.2f} s")
    print(f"  Requests           : {summary['requests']}  "
          f"({summary['successful']} ok / {summary['failed']} failed)")
    print()
    print(f"  TPS (all)          : {summary['tps']:.2f} req/s")
    print(f"  TPS (2xx/3xx only) : {summary['tps_successful']:.2f} req/s")
    print(f"  Throughput         : {human_bytes(summary['throughput_bytes_per_s'])}/s")
    print()
    print("  Latency (ms)")
    print(f"    min {L['min']:>9.2f}   avg {L['avg']:>9.2f}   max {L['max']:>9.2f}")
    print(f"    p50 {L['p50']:>9.2f}   p90 {L['p90']:>9.2f}   "
          f"p95 {L['p95']:>9.2f}   p99 {L['p99']:>9.2f}")
    if summary["status_codes"]:
        print()
        print("  Status codes")
        for code, count in summary["status_codes"].items():
            print(f"    {code:<6} {count}")
    if summary["errors"]:
        print()
        print("  Errors")
        for name, count in summary["errors"].items():
            print(f"    {count:>6}  {name}")
    print("=" * 62)
    return summary


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tps.py",
        description="Measure a website's TPS with a customizable number of workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Only load-test systems you own or are authorized to test.",
    )
    p.add_argument("url", help="target URL, e.g. https://example.com/health")
    p.add_argument("-w", "--workers", type=int, default=10,
                   help="number of concurrent workers (default: 10)")
    p.add_argument("-d", "--duration", type=float, default=10.0,
                   help="seconds to run (default: 10); ignored if -n is given")
    p.add_argument("-n", "--requests", type=int,
                   help="total requests to send instead of running for a duration")
    p.add_argument("-X", "--method", default="GET", help="HTTP method (default: GET)")
    p.add_argument("-H", "--header", action="append", default=[], metavar="'K: V'",
                   help="extra request header; repeatable")
    p.add_argument("-b", "--body", help="request body string")
    p.add_argument("--body-file", help="read request body from a file")
    p.add_argument("-t", "--timeout", type=float, default=10.0,
                   help="per-request timeout in seconds (default: 10)")
    p.add_argument("--rps", type=float,
                   help="cap offered load at this many requests/sec (open model)")
    p.add_argument("--ramp-up", type=float, default=0.0, metavar="SEC",
                   help="stagger worker start-up over this many seconds")
    p.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="discard the first N requests from the stats")
    p.add_argument("-k", "--insecure", action="store_true",
                   help="skip TLS certificate verification")
    p.add_argument("--interval", type=float, default=1.0,
                   help="live progress refresh in seconds, 0 to disable (default: 1)")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON only")

    args = p.parse_args(argv)
    if "://" not in args.url:
        args.url = "https://" + args.url
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.requests is not None and args.requests < 1:
        p.error("--requests must be >= 1")
    if args.requests is None and args.duration <= 0:
        p.error("--duration must be > 0")
    if args.body and args.body_file:
        p.error("use either --body or --body-file, not both")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    stats = Stats()
    budget = Budget(args.requests)
    limiter = RateLimiter(args.rps) if args.rps else None
    mode = f"{args.requests} requests" if args.requests else f"{args.duration:g}s"
    print(f"Load testing {args.method.upper()} {args.url}\n"
          f"  {args.workers} workers, {mode}"
          + (f", capped at {args.rps:g} rps" if limiter else "")
          + (f", ramp-up {args.ramp_up:g}s" if args.ramp_up else ""), file=sys.stderr)

    step = (args.ramp_up / args.workers) if args.ramp_up else 0.0
    threads = [
        threading.Thread(target=worker, args=(i, target, stats, budget, limiter, i * step),
                         daemon=True, name=f"w{i}")
        for i in range(args.workers)
    ]

    stats.started = time.perf_counter()
    deadline = stats.started + args.duration + args.ramp_up if args.requests is None else None

    reporter = None
    if args.interval > 0 and not args.json:
        reporter = threading.Thread(target=progress,
                                    args=(stats, deadline, budget, args.interval),
                                    daemon=True)

    for t in threads:
        t.start()
    if reporter:
        reporter.start()

    try:
        if deadline is not None:
            STOP.wait(max(0.0, deadline - time.perf_counter()))
            STOP.set()
        else:
            for t in threads:
                while t.is_alive():
                    t.join(0.2)
            STOP.set()
    except KeyboardInterrupt:
        STOP.set()
        print("\ninterrupted - reporting what we have", file=sys.stderr)

    for t in threads:
        t.join(args.timeout + 1)
    stats.finished = time.perf_counter()
    if reporter:
        reporter.join(0.5)
        sys.stderr.write("\r" + " " * 78 + "\r")
        sys.stderr.flush()

    if stats.total == 0:
        print("error: no requests completed", file=sys.stderr)
        return 1

    # Warm-up only trims the latency distribution (DNS, TLS, cold caches);
    # request/status counts still reflect every transaction that was sent.
    latencies = stats.latencies
    if args.warmup and len(latencies) > args.warmup:
        latencies = latencies[args.warmup:]

    summary = report(stats, args, latencies)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
