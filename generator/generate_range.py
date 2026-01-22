import argparse
from datetime import datetime, timedelta, timezone

from . import __version__ as GENERATOR_VERSION
from .config import GIT_COMMIT
from .generate_one_day import generate_unique, build_meta
from .export_json import write_json, strip_internal


def iso_date(d) -> str:
    return d.strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--timezone", default="UTC")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    available_through = today + timedelta(days=args.days - 1)

    for offset in range(args.days):
        d = today + timedelta(days=offset)
        date_utc = iso_date(d)

        puzzles = {
            "easy": generate_unique(date_utc, "easy"),
            "medium": generate_unique(date_utc, "medium"),
            "hard": generate_unique(date_utc, "hard"),
        }

        meta = build_meta(date_utc, puzzles)
        base_dir = f"public/api/daily/{date_utc}"

        write_json(f"{base_dir}/meta.json", meta)
        for diff in ("easy", "medium", "hard"):
            write_json(f"{base_dir}/{diff}.json", strip_internal(puzzles[diff]))

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


if __name__ == "__main__":
    main()
