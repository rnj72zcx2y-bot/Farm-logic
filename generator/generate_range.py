import argparse
from datetime import datetime, timedelta, timezone

from .generate_one_day import generate_unique, build_meta
from .export_json import write_json, strip_internal
from . import __version__ as GENERATOR_VERSION
from .config import GIT_COMMIT


def iso_date(d) -> str:
    return d.strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--timezone", default="UTC")
    # keep flag for compatibility, but logs are always on in this dev version
    ap.add_argument("--verbose", action="store_true", help="(ignored) logs are always printed")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    available_through = today + timedelta(days=args.days - 1)

    diffs = ("easy", "medium", "hard")

    for offset in range(args.days):
        d = today + timedelta(days=offset)
        date_utc = iso_date(d)

        print(f"[GEN] date={date_utc} start", flush=True)

        puzzles = {}
        for diff in diffs:
            print(f"[GEN] date={date_utc} diff={diff} ...", flush=True)
            puzzles[diff] = generate_unique(date_utc, diff)
            print(f"[GEN] date={date_utc} diff={diff} ✓", flush=True)

        meta = build_meta(date_utc, puzzles)
        base_dir = f"public/api/daily/{date_utc}"

        write_json(f"{base_dir}/meta.json", meta)
        for diff in diffs:
            write_json(f"{base_dir}/{diff}.json", strip_internal(puzzles[diff]))

        print(f"[GEN] date={date_utc} written", flush=True)

    today_str = iso_date(today)
    write_json("public/api/today.json", {
        "dateUtc": today_str,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "gitCommit": GIT_COMMIT,
        "paths": {
            "easy": f"/api/daily/{today_str}/easy.json",
            "medium": f"/api/daily/{today_str}/medium.json",
            "hard": f"/api/daily/{today_str}/hard.json",
            "meta": f"/api/daily/{today_str}/meta.json"
        }
    })

    write_json("public/api/latest.json", {
        "generatedAtUtc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "availableThroughUtc": iso_date(available_through),
        "daysAhead": args.days,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "gitCommit": GIT_COMMIT
    })

    print(f"[GEN] done. availableThroughUtc={iso_date(available_through)}", flush=True)


if __name__ == "__main__":
    main()
