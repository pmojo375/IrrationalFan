# Irrational Fan

Discover technically accurate, deliberately cherry-picked college-football stories.
Reports run in the console or can be saved as text files.

## Build and refresh the data

```bash
python -m pip install -r requirements.txt
python process.py
```

Processing automatically downloads missing source Parquets and refreshes the active
football season’s schedule, weekly summaries and season-specific team metadata.
The active season runs August through July, so January bowls refresh the prior
calendar year’s files. Existing historical files are reused. Sources are SportsDataverse release assets:

| Source | Saved under | Used for |
| --- | --- | --- |
| `cfb_schedules` | `data/schedules/` | Completed results, upcoming matchups, displayed poll ranks, game-time conferences |
| `cfb_team_summaries_weekly` | `data/summaries/` | Existing prior-week team and opponent statistics |
| `espn_cfb_teams` | `data/teams/` | Team-ID/season conference fallback, respecting realignment |

Each newly downloaded source has a `.source.json` sidecar containing its URL,
fetch time and dimensions. Downloads are validated and atomically replace the
cached source; a failed refresh aborts processing and preserves the previous file.
An unpublished historical team file (HTTP 404) is optional: schedule and summary
conference values are still used. Other download errors are failures.

```bash
python process.py --offline         # rebuild only from saved files
python process.py --force-download  # refresh historical files too
```

`--download` remains a compatibility alias for the normal online build.
The main output is `cfb_team_games.parquet`. No extra package or play-by-play
requests are needed for this feature set. The manual matchup Action refreshes
sources before processing; automated PR tests use the checked-in data offline.

## New searchable context

Both team and opponent perspectives are stored in the main table. Opponent
columns use the `opp_` prefix.

- `previous_points_for`, `previous_points_against`, `previous_margin`,
  `previous_won`, `previous_conference_game`, `previous_opponent_conference`:
  the immediately previous completed game in the same season.
- `rest_days`: full calendar days between game dates, measured in US Eastern
  time, excluding both game dates. Saturday to Saturday is **6**.
- `after_bye`: at least **12 calendar days** between games. This is an explicit
  gap rule, not a claim that the league officially designated a bye week.
- `rank`: the schedule provider's displayed pregame Top-25 poll rank. The poll
  can change from AP to CFP (historically BCS) during a season. Missing ranks
  remain unknown and do not become rank 0, 26 or a confirmed unranked flag.
- `conference`, `conference_competition`, `conference_wins`, `conference_losses`:
  season-specific membership, the schedule's explicit conference-game flag,
  and conference results strictly before the game. Unknown historical flags
  leave subsequent conference totals unknown rather than silently counting 0.

Previous-game/rest features are missing at the first game of a season; they do
not carry over from the offseason. Conference records reset each season.
Conference-name aliases are normalized, but historical membership is not changed.

Upcoming discovery reads the saved schedule's current rank, week, conference and
neutral-site information, then applies the same context calculation to completed
history plus an outcome-free matchup row. A previous game's rank is never used
as the upcoming game's rank. Upcoming ranks describe the **latest downloaded
snapshot**, so refresh before running. Historical ranks describe the provider's
historical kickoff snapshot. Current-game scores and outcomes never become inputs.

## Run discovery

```bash
python -m discovery.main --team Nevada --opponent UCLA --date 2026-10-31 \
  --data cfb_team_games.parquet --output matchup_report.txt
python -m pytest -q
```

The date must identify an upcoming game in the saved schedule. New fields are
included in default discovery, with readable condition labels and a current-form
snapshot. Atomic candidate selection gives available features a turn before
spending the remaining budget on threshold variants. The existing quality,
evidence-overlap and streak-diversity filters still apply; a short report is valid.

If there is an unplayed scheduled game before the requested matchup, its score
and outcome remain unknown: previous-game result features are withheld, rest is
calculated from its scheduled date, and conference totals are withheld if that
pending game can change them. Thus looking two weeks ahead does not invent a bye.
