"""Human-readable discovery statements for college football."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from discovery.engine.models import Condition, Discovery, Operator
from discovery.sports.college_football.features import MONTH_NAMES


SCOPE_LABELS = {
    "team": "{team} History",
    "opponent": "Opponent History",
    "all": "All College Football",
    "h2h": "Head-to-Head",
    "both": "Both Teams",
}


def scope_label(scope: str, team: str = "", opponent: str = "") -> str:
    if scope == "team":
        return f"{team} History"
    if scope == "opponent":
        return f"{opponent} History"
    if scope == "h2h":
        return f"{team} vs {opponent} Head-to-Head"
    return SCOPE_LABELS.get(scope, scope)


def condition_phrase(cond: Condition) -> str:
    """Short bullet-friendly phrase for a condition."""
    feat = cond.feature
    op = cond.operator
    val = cond.value

    if feat == "is_home":
        return "Home game" if val else "Road game"
    if feat == "neutral_site":
        return "Neutral-site game" if val else "Non-neutral site"
    if feat == "conference_competition":
        return "Conference game" if val else "Non-conference game"
    if feat in ("month", "month_name"):
        name = MONTH_NAMES.get(int(val), val) if feat == "month" else val
        return f"Month: {name}"
    if feat == "day_of_week":
        return f"Day: {val}"
    if feat == "week":
        return _numeric_phrase("Week", op, val)
    if feat == "wins":
        return _numeric_phrase("Pregame wins", op, val)
    if feat == "losses":
        return _numeric_phrase("Pregame losses", op, val)
    if feat == "games_played":
        return _numeric_phrase("Games played", op, val)
    if feat == "win_pct":
        return _numeric_phrase("Pregame win percentage", op, val, pct=True)
    if feat == "win_streak":
        return _numeric_phrase("Win streak", op, val)
    if feat == "loss_streak":
        return _numeric_phrase("Loss streak", op, val)
    if feat == "has_winning_record":
        return "Entering with a winning record" if val else "Not entering with a winning record"
    if feat == "has_losing_record":
        return "Entering with a losing record" if val else "Not entering with a losing record"
    if feat == "opp_wins":
        return _numeric_phrase("Opponent pregame wins", op, val)
    if feat == "opp_losses":
        return _numeric_phrase("Opponent pregame losses", op, val)
    if feat == "opp_games_played":
        return _numeric_phrase("Opponent games played", op, val)
    if feat == "opp_win_pct":
        return _numeric_phrase("Opponent win percentage", op, val, pct=True)
    if feat == "opp_win_streak":
        return _numeric_phrase("Opponent win streak", op, val)
    if feat == "opp_loss_streak":
        return _numeric_phrase("Opponent loss streak", op, val)
    if feat == "opp_has_winning_record":
        return "Opponent entering with a winning record" if val else "Opponent not entering with a winning record"
    if feat == "opp_has_losing_record":
        return "Opponent entering with a losing record" if val else "Opponent not entering with a losing record"
    if feat == "win_diff":
        return _numeric_phrase("Win differential vs opponent", op, val)
    if feat == "season_type":
        return f"Season type: {val}"
    return cond.display()


def _numeric_phrase(label: str, op: Operator, val: Any, pct: bool = False) -> str:
    if pct:
        try:
            shown = f"{float(val):.0%}"
        except (TypeError, ValueError):
            shown = val
    else:
        shown = val
    if op == Operator.EQ:
        return f"{label} == {shown}"
    if op == Operator.GE:
        return f"{label} >= {shown}"
    if op == Operator.LE:
        return f"{label} <= {shown}"
    if op == Operator.GT:
        return f"{label} > {shown}"
    if op == Operator.LT:
        return f"{label} < {shown}"
    return f"{label} {op.value} {shown}"


def describe_discovery(
    discovery: Discovery,
    *,
    team: str,
    opponent: str,
) -> str:
    """Build a prose claim that matches the scope, objective, and record."""
    snippet = _condition_snippet(discovery.conditions)

    if discovery.scope == "team":
        subject = team
        if discovery.objective == "streak":
            return (
                f"{subject} has won its last {discovery.streak} qualifying games"
                f"{snippet}. "
                f"Overall qualifying record: {discovery.wins}-{discovery.losses}."
            )
        return f"{subject} is {discovery.wins}-{discovery.losses}{snippet}."

    if discovery.scope == "opponent":
        subject = opponent
        if discovery.objective == "streak":
            return (
                f"{subject} has lost its last {discovery.streak} qualifying games"
                f"{snippet}. "
                f"Overall qualifying record in this sample: "
                f"{discovery.losses}-{discovery.wins} (W-L from {subject}'s perspective)."
            )
        # favorable = losses for opponent scope: wins field counts favorable (losses)
        opp_losses = discovery.wins
        opp_wins = discovery.losses
        return (
            f"{subject} is {opp_wins}-{opp_losses}{snippet} "
            f"- a historically unfavorable spot for {subject}."
        )

    if discovery.scope == "h2h":
        if discovery.objective == "streak":
            return (
                f"{team} has won its last {discovery.streak} meetings with {opponent}"
                f"{snippet}. "
                f"Overall qualifying H2H record: {discovery.wins}-{discovery.losses}."
            )
        return f"{team} is {discovery.wins}-{discovery.losses} against {opponent}{snippet}."

    # all college football
    if discovery.objective == "streak":
        return (
            f"Teams matching these conditions have won their last "
            f"{discovery.streak} qualifying games{snippet}. "
            f"Overall qualifying record: {discovery.wins}-{discovery.losses}."
        )
    return f"Teams matching these conditions are {discovery.wins}-{discovery.losses}{snippet}."


def _condition_snippet(conditions: Sequence[Condition]) -> str:
    if not conditions:
        return ""
    phrases = [condition_phrase(c) for c in conditions]
    if len(phrases) <= 3:
        return " when " + "; ".join(phrases)
    return " when " + "; ".join(phrases[:3]) + f"; (+{len(phrases) - 3} more)"


def format_evidence_line(game) -> str:
    season = game.season if game.season is not None else ""
    date = _short_date(game.date)
    prefix = f"{season}" if season != "" else date
    ts = game.team_score if game.team_score is not None else "?"
    os_ = game.opponent_score if game.opponent_score is not None else "?"
    return f"  {prefix}: {game.team} {ts}, {game.opponent} {os_}"


def _short_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def format_report(
    *,
    team: str,
    opponent: str,
    game_date: str,
    home: bool,
    discoveries: Sequence[Discovery],
    stats: Optional[Any] = None,
) -> str:
    """Console / text-file report."""
    lines: list[str] = []
    lines.append("=" * 52)
    lines.append("IRRATIONAL FAN")
    lines.append(f"{team} vs {opponent}")
    lines.append(_pretty_date(game_date))
    location = "Home" if home else "Away"
    lines.append(f"Site: {location} for {team}")
    lines.append("=" * 52)

    if not discoveries:
        lines.append("")
        lines.append("No discoveries met the current search thresholds.")
    else:
        for i, disc in enumerate(discoveries, start=1):
            lines.append("")
            lines.append(f"DISCOVERY #{i}")
            lines.append(f"Scope: {scope_label(disc.scope, team, opponent)}")
            lines.append(f"Type: {'Streak' if disc.objective == 'streak' else 'Record'}")
            lines.append("")
            desc = disc.description or describe_discovery(disc, team=team, opponent=opponent)
            # Wrap description lightly
            lines.append(desc)
            lines.append("")
            lines.append("Conditions:")
            if disc.conditions:
                for c in disc.conditions:
                    lines.append(f"  - {condition_phrase(c)}")
            else:
                lines.append("  - (none)")
            lines.append("")
            if disc.favorable_label == "losses":
                lines.append(
                    f"Favorable Outcomes (losses for {opponent}): {disc.wins}"
                )
                lines.append(f"Unfavorable Outcomes: {disc.losses}")
                lines.append(
                    f"Opponent Qualifying W-L: {disc.losses}-{disc.wins}"
                )
            else:
                lines.append(f"Historical Record: {disc.wins}-{disc.losses}")
            lines.append(f"Success Rate: {disc.win_pct:.0%}")
            lines.append(f"Matching Games: {disc.n_games}")
            if disc.streak:
                lines.append(f"Current Qualifying Streak: {disc.streak}")
            lines.append(f"Ranking Score: {disc.ranking_score:.1f}")
            lines.append("")
            lines.append("Historical Evidence:")
            evidence = disc.matching_games
            show = evidence if len(evidence) <= 12 else evidence[:6] + evidence[-3:]
            if len(evidence) > 12:
                for g in evidence[:6]:
                    lines.append(format_evidence_line(g))
                lines.append(f"  ... ({len(evidence) - 9} more) ...")
                for g in evidence[-3:]:
                    lines.append(format_evidence_line(g))
            else:
                for g in show:
                    lines.append(format_evidence_line(g))
            lines.append("")
            lines.append("-" * 52)

    lines.append("")
    if stats is not None:
        lines.append(
            f"Search: {stats.candidates_evaluated} candidates evaluated in "
            f"{stats.elapsed_seconds:.2f}s "
            f"({stats.discoveries_before_dedup} raw -> {stats.discoveries_after_dedup} kept)"
        )
    lines.append("")
    lines.append(
        "Note: These are cherry-picked historical patterns, not statistically "
        "validated predictions."
    )
    lines.append("=" * 52)
    return "\n".join(lines)


def _pretty_date(game_date: str) -> str:
    try:
        import pandas as pd

        dt = pd.to_datetime(game_date)
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        return str(game_date)
