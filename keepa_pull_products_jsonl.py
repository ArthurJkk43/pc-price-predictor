import argparse
import inspect
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

import keepa


def json_default(o: Any):
    import datetime as dt

    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, dt.timedelta):
        return o.total_seconds()

    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    try:
        import pandas as pd
        if isinstance(o, pd.Timestamp):
            return None if pd.isna(o) else o.isoformat()
    except Exception:
        pass

    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")

    if isinstance(o, set):
        return list(o)

    return str(o)


def read_asins_txt(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a = line.strip().upper()
            if len(a) == 10:
                out.append(a)
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


def load_done_asins_from_jsonl(path: str) -> Set[str]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                asin = obj.get("asin")
                if isinstance(asin, str) and len(asin) == 10:
                    done.add(asin.upper())
            except Exception:
                continue
    return done


def build_query_kwargs(api: keepa.Keepa, domain: str, history: bool, days: int) -> Dict[str, Any]:
    sig = inspect.signature(api.query)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    if "domain" in params:
        kwargs["domain"] = domain
    if "history" in params:
        kwargs["history"] = history
    if "wait" in params:
        kwargs["wait"] = True

    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if "days" in params or has_varkw:
        kwargs["days"] = days

    return kwargs


def query_with_retry(api: keepa.Keepa, asins: List[str], kwargs: Dict[str, Any], retries: int, backoff: float):
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            res = api.query(asins, **kwargs)
            if isinstance(res, list):
                return [p for p in res if isinstance(p, dict)]
            return []
        except Exception as e:
            last_err = e
            sleep_s = backoff * (2 ** attempt)
            print(f"[WARN] query failed (attempt {attempt+1}/{retries+1}): {e} | sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"query failed after retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asins", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--domain", default="US")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--days", type=int, default=2190)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=2.0)
    args = ap.parse_args()

    key = os.environ.get("KEEPA_KEY")
    if not key:
        raise SystemExit('Missing KEEPA_KEY env var.')

    api = keepa.Keepa(key)

    asins = read_asins_txt(args.asins)
    if not asins:
        raise SystemExit(f"No ASINs found in {args.asins}")

    done = set()
    if args.resume:
        done = load_done_asins_from_jsonl(args.out_jsonl)
        print(f"[INFO] Resume: already have {len(done)} products")

    pending = [a for a in asins if a not in done]
    print(f"[INFO] Total ASINs: {len(asins)} | Pending: {len(pending)}")

    err_path = args.out_jsonl + ".errors.jsonl"
    q_kwargs = build_query_kwargs(api, domain=args.domain, history=args.history, days=args.days)

    with open(args.out_jsonl, "a", encoding="utf-8") as out, open(err_path, "a", encoding="utf-8") as err:
        batch_no = 0
        for i in range(0, len(pending), args.batch):
            batch_no += 1
            chunk = pending[i:i + args.batch]
            try:
                prods = query_with_retry(api, chunk, q_kwargs, retries=args.retries, backoff=args.backoff)
            except Exception as e:
                err.write(json.dumps({"batch": batch_no, "asins": chunk, "error": str(e)}, ensure_ascii=False) + "\n")
                err.flush()
                time.sleep(args.delay)
                continue

            written = 0
            for p in prods:
                asin = p.get("asin")
                if isinstance(asin, str) and len(asin) == 10:
                    p["asin"] = asin.upper()
                out.write(json.dumps(p, ensure_ascii=False, default=json_default) + "\n")
                written += 1

            out.flush()
            print(f"[OK] batch {batch_no}: requested={len(chunk)} returned={written}")
            time.sleep(args.delay)

    print(f"[DONE] output: {args.out_jsonl}")
    print(f"[DONE] errors: {err_path}")


if __name__ == "__main__":
    main()