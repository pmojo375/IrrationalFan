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

## Web interface (FastAPI)

Install the optional web dependencies and generate the processed data:

```bash
python -m pip install -r requirements-web.txt
python process.py
uvicorn webapp.app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/ locally. The page lists the earliest upcoming
scheduled season/week, with a separate selector for all future scheduled games.
Choose a matchup and the team whose case to argue; the server calls the same
`discover()` engine as the CLI and displays its full text report.

### Background discoveries and recent history

The page starts a search with `POST /api/jobs` and immediately receives a job ID
(HTTP 202), then polls `GET /api/jobs/{id}` until it is complete or failed.
Queued jobs include a queue position; running jobs include their start time.
Completed job details include the report, structured discoveries and original settings.
`GET /api/jobs` lists recent searches without downloading every report.
An identical active search reconnects to its existing job; an identical completed
search returns its saved result (HTTP 200). Different searches queue behind one
worker. There are at most eight unfinished jobs; additional searches receive
HTTP 429 with `Retry-After: 10`. Duplicate searches can still reconnect when full.
Invalid settings are rejected before a job is queued.

The page remembers the current job locally and reconnects after a refresh, including
when the initial submit response was lost. It retries temporary connection errors
and automatically displays finished results. Recent discoveries shows the style,
creation time and status; opening one restores the search settings. Failed searches
have a retry button. Clear finished discoveries removes only completed/failed jobs,
leaving queued and running work alone.

Jobs and results live in `data/web_jobs.sqlite3` by default. To place this file in
another persistent, writable directory, set `IRRATIONALFAN_JOB_DB=/path/to/jobs.sqlite3`
in the service environment. No new package, broker or separate worker service is
required. The latest 32 finished searches are retained alongside unfinished work.
Completed results survive service restarts. Work interrupted by a forced restart
is marked failed with a retry message; it is not silently reported as still running.
New requests invalidate saved-result reuse when the processed data, relevant
schedules, effective configuration or application version changes. Existing history
remains available as a snapshot. Bump the application version when changing engine
semantics without a data/configuration change.

**Run Uvicorn with one worker.** A process lock prevents a second worker from
opening the same job store. Jobs run on one dedicated thread independently of
HTTP connections; starting more web workers would duplicate the engine's data cache
and increase memory use. Normal shutdown waits for accepted work; after a forced
shutdown, reopen Recent discoveries to retry interrupted jobs.

Compatibility endpoints remain: `POST /api/discover` waits for a report and can
still exceed proxy timeouts, `GET /api/cache` returns saved completed reports,
and `DELETE /api/cache` clears finished history. The browser uses the job endpoints.
`GET /api/games?mode=week|future&team=...` and `GET /health` are unchanged.
Game IDs and selected teams are checked against the saved schedule. Today's games
are eligible, including in-progress games; previously completed, canceled and
postponed games are excluded when the saved schedule marks them accordingly.

This app reads cached source files; it does **not** refresh schedules on web
requests. Run `python process.py` periodically (for example daily) as a separate
job. Keep the Uvicorn listener on localhost behind Nginx, and apply rate limiting
to the potentially CPU-heavy discovery endpoints before public deployment.
When proxying at a subdomain, route the entire hostname to port 8001. Run
`python -m pytest -q` to check the engine, API and persistent job queue.

### Individual stat controls

Customize searchable stats offers a searchable, categorized list with separate
team and opponent entries. Each stat has Allow, Require and Exclude states, plus
category-wide Allow/Exclude controls. Advanced metrics start excluded and become
searchable when explicitly allowed. Dataset-missing advanced metrics are disabled.
Preferences are remembered in the browser; opening recent discoveries restores
its original selections. New team-color and stadium-capacity features remain disabled.

“Search only stats I explicitly select” limits the search to stats the visitor
has explicitly changed to Allow or Require. Otherwise all allowed stats can be
used. Required stats must appear in every returned discovery; excluded stats never
enter candidate generation. Required stats seed beam expansion before optional
conditions, including when another required condition leaves the evidence set
unchanged. Requirements unavailable in the matchup return an actionable message;
valid searches with insufficient qualifying history can still return zero results.
Requirements are never relaxed automatically. Search remains a bounded heuristic,
so a zero-result search is not proof that no mathematical combination exists.

`GET /api/features` provides the curated stat catalog and dataset availability.
Job submissions accept `allowed_stats` (a whitelist, or null for style/category
defaults), `required_stats`, and `excluded_stats`. Unknown IDs, duplicates,
required/excluded conflicts, or too many requirements are rejected. Existing
category-based API requests remain supported. For opponent-history searches,
team/opponent stat selections are mapped to that scope's underlying perspective.
Saved-job identity includes the stat selections but ignores their order and purely
visual picker settings.
