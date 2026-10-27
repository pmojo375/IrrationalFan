"""CLI entry point for Irrational Fan discovery runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from discovery.sports.college_football.matchup import discover


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover cherry-picked historical CFB statistics for a matchup."
    )
    p.add_argument("--team", required=True, help='Selected team, e.g. "Michigan State"')
    p.add_argument("--opponent", required=True, help='Opponent, e.g. "Michigan"')
    p.add_argument("--date", required=True, help="Upcoming game date YYYY-MM-DD")
    p.add_argument("--home", action="store_true", default=False, help="Selected team is at home")
    p.add_argument("--away", action="store_true", default=False, help="Selected team is on the road")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--min-sample", type=int, default=None)
    p.add_argument("--max-conditions", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument(
        "--scopes",
        default="team,opponent,h2h",
        help="Comma-separated scopes: team,opponent,h2h,all,both",
    )
    p.add_argument("--output", type=str, default=None, help="Optional .txt output path")
    p.add_argument("--data", type=str, default=None, help="Path to team-games parquet")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.home and args.away:
        raise SystemExit("Specify only one of --home or --away")
    home = True if args.home else False if args.away else None

    scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())
    discover(
        team=args.team,
        opponent=args.opponent,
        game_date=args.date,
        home=home,
        top_n=args.top_n,
        scopes=scopes,
        min_sample_size=args.min_sample,
        max_conditions=args.max_conditions,
        beam_width=args.beam_width,
        data_path=Path(args.data) if args.data else None,
        output_path=Path(args.output) if args.output else None,
        print_report=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
