# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic>=0.92",
#     "openai>=2.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""Do production compaction endpoints match a Bloom filter on set membership?

Records a set of malicious URLs in a conversation, compacts it with a
production endpoint (Claude or OpenAI), asks membership queries on the
compacted output, and compares size and error rates against the Bloom
filter's analytic frontier. See README.md.

Usage:
    uv run experiment.py run --arm claude --n 20000
    uv run experiment.py plot

`run` appends one JSON record to results/runs.jsonl. `plot` writes the
figure and the false-negative table to results/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import matplotlib
import numpy as np
from openai import OpenAI

matplotlib.use("Agg")  # must come before the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

log = logging.getLogger("experiment")

# Fable 5 refuses bulk URL lists (cyber safeguard); Opus 4.8 is newest that doesn't.
CLAUDE_MODEL = "claude-opus-4-8"
OPENAI_MODEL = "gpt-5.5"
CLAUDE_COMPACTION_BETA = "compact-2026-01-12"
MIN_TRIGGER_TOKENS = 50_000  # documented minimum for the Claude trigger

# Items go in the assistant turn (OpenAI compaction keeps user turns
# verbatim); the conversation must end on a user turn (a trailing
# assistant turn is a rejected prefill).
SCAN_PROMPT = (
    "You have recorded a set of strings. List every string in the set, "
    "one per line."
)
DONE_PROMPT = (
    "The set is complete. Later in this conversation you will be asked "
    "whether a given string was or was not in the set. Reply with OK."
)
QUERY_PROMPT = (
    "Earlier in this conversation you listed a set of strings. Was the "
    "following string in the set?\n\nString: {item}"
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

COMPACTION_INSTRUCTIONS = (
    "You are compacting this conversation. Afterward the original messages "
    "are discarded, and you must answer set membership queries using only "
    "your compacted summary: for a given string, whether it was among the "
    "strings listed above. Compact the conversation so as to minimize the "
    "number of membership queries you answer incorrectly. Use whatever "
    "representation best achieves this."
)

CONTEXT_MANAGEMENT = {
    "edits": [
        {
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": MIN_TRIGGER_TOKENS},
            "instructions": COMPACTION_INSTRUCTIONS,
        }
    ]
}
COMPACTION_MAX_TOKENS = 16000
QUERY_MAX_TOKENS = 2048
CLAUDE_OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
OPENAI_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "membership_answer",
        "schema": ANSWER_SCHEMA,
        "strict": True,
    }
}
FAIL_FAST_QUERIES = 5
QUERY_WORKERS = 4
QUERY_TIMEOUT_S = 600  # wall-clock cap so a rate-limit stall can't wedge a run


def bloom_fpr(bits_per_item: float) -> float:
    """Optimal-k Bloom filter (Broder and Mitzenmacher 2004)."""
    return math.pow(2.0, -bits_per_item * math.log(2.0))


def lower_bound_fpr(bits_per_item: float) -> float:
    """Minimum for any tester with no false negatives (Carter et al. 1978)."""
    return math.pow(2.0, -bits_per_item)


def url_dataset(
    n: int, num_non_members: int, seed: int, csv_path: Path
) -> tuple[list, list]:
    """Members and held-out non-members from the URL CSV. Drawing
    non-members from the same pool means membership cannot be guessed from
    the surface form of a URL."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; see data/README.md.")
    seen: set[str] = set()
    urls: list[str] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            url = row["url"].strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    if len(urls) < n + num_non_members:
        raise ValueError(f"need {n + num_non_members} URLs, have {len(urls)}")
    rng = random.Random(seed)
    rng.shuffle(urls)
    return urls[:n], urls[n : n + num_non_members]


def gzip_bits(text: str) -> int:
    """gzip size: a model-agnostic axis for summaries and filters alike."""
    return 8 * len(gzip.compress(text.encode("utf-8"), 9))


def conversation(members: list[str]) -> list[dict]:
    return [
        {"role": "user", "content": SCAN_PROMPT},
        {"role": "assistant", "content": "\n".join(members)},
        {"role": "user", "content": DONE_PROMPT},
    ]


def make_queries(
    members: list[str], non_members: list[str], num_queries: int, seed: int
) -> tuple[list, list]:
    """Half the queries probe false negatives, half false positives."""
    rng = random.Random(seed)
    half = num_queries // 2
    return rng.sample(members, half), rng.sample(non_members, half)


def run_queries(
    ask: Callable[[str], str | None],
    member_qs: list[str],
    non_member_qs: list[str],
) -> tuple[list, list]:
    """Answer all queries, the bulk concurrently. A short sequential probe
    aborts on a systematic failure before the rest are dispatched."""
    items = member_qs + non_member_qs
    log.info("answering %d membership queries", len(items))
    probe = [ask(item) for item in items[:FAIL_FAST_QUERIES]]
    if all(a is None for a in probe):
        raise SystemExit(
            f"First {FAIL_FAST_QUERIES} queries all failed; aborting run."
        )

    rest = items[FAIL_FAST_QUERIES:]
    answers = probe + [None] * len(rest)
    done = 0
    ex = ThreadPoolExecutor(max_workers=QUERY_WORKERS)
    futures = {ex.submit(ask, item): i for i, item in enumerate(rest)}
    try:
        for fut in as_completed(futures, timeout=QUERY_TIMEOUT_S):
            answers[FAIL_FAST_QUERIES + futures[fut]] = fut.result()
            done += 1
            if done % 25 == 0:
                log.info("  %d/%d queries done", done, len(rest))
    except TimeoutError:
        log.warning("query phase exceeded %ds; %d/%d returned, rest count as failed",
                    QUERY_TIMEOUT_S, done, len(rest))
    ex.shutdown(wait=False, cancel_futures=True)
    return answers[: len(member_qs)], answers[len(member_qs) :]


def score(member_answers: list, non_member_answers: list) -> dict:
    """A failed query yields None, which counts as a wrong answer."""
    fn = sum(1 for a in member_answers if a != "YES")
    fp = sum(1 for a in non_member_answers if a == "YES")
    total = len(member_answers) + len(non_member_answers)
    return {
        "num_members_queried": len(member_answers),
        "num_non_members_queried": len(non_member_answers),
        "error_rate": (fn + fp) / total,
        "false_negative_rate": fn / len(member_answers),
        "false_positive_rate": fp / len(non_member_answers),
        "num_failed_queries": sum(
            1 for a in member_answers + non_member_answers if a is None
        ),
    }


def run_claude(
    members: list[str],
    member_qs: list[str],
    non_member_qs: list[str],
    model: str,
) -> dict:
    """Claude server-side compaction with membership-preserving instructions.
    https://platform.claude.com/docs/en/build-with-claude/compaction
    """
    # Bounded timeout: the compaction call occasionally stalls for tens of minutes.
    client = anthropic.Anthropic(timeout=300.0, max_retries=3)
    messages = conversation(members)

    count = client.beta.messages.count_tokens(
        betas=[CLAUDE_COMPACTION_BETA],
        model=model,
        messages=messages,
        context_management=CONTEXT_MANAGEMENT,
    )
    if count.input_tokens <= MIN_TRIGGER_TOKENS:
        raise SystemExit(
            f"Conversation is {count.input_tokens} tokens, below the "
            f"{MIN_TRIGGER_TOKENS}-token compaction trigger. Increase --n."
        )

    log.info("compacting %d items (%d tokens) with %s", len(members),
             count.input_tokens, model)
    response = client.beta.messages.create(
        betas=[CLAUDE_COMPACTION_BETA],
        model=model,
        max_tokens=COMPACTION_MAX_TOKENS,
        messages=messages,
        context_management=CONTEXT_MANAGEMENT,
    )
    if response.stop_reason == "max_tokens":
        raise SystemExit(
            "Compaction response hit max_tokens; the summary may be "
            "truncated. Raise COMPACTION_MAX_TOKENS."
        )
    if response.stop_reason == "refusal":
        raise SystemExit(
            f"Model refused to compact: {getattr(response, 'stop_details', None)}"
        )
    summaries = [b.content for b in response.content if b.type == "compaction"]
    if not summaries or summaries[0] is None:
        raise SystemExit("Compaction did not trigger or produced no summary.")
    summary = summaries[0]
    if not isinstance(summary, str):  # content may be a list of text blocks
        summary = "".join(getattr(part, "text", "") for part in summary)
    log.info("compacted: summary %d chars (%.2f bits/item)",
             len(summary), gzip_bits(summary) / len(members))
    log.info("compacted summary:\n%s", summary)

    # Query from the compaction block alone; the pre-compaction turns are
    # dropped server-side and not re-billed.
    compacted_turn = {
        "role": "assistant",
        "content": [b.model_dump(exclude_none=True) for b in response.content],
    }

    def ask(item: str) -> str | None:
        try:
            reply = client.beta.messages.create(
                betas=[CLAUDE_COMPACTION_BETA],
                model=model,
                max_tokens=QUERY_MAX_TOKENS,
                messages=[
                    compacted_turn,
                    {"role": "user", "content": QUERY_PROMPT.format(item=item)},
                ],
                context_management=CONTEXT_MANAGEMENT,
                output_config=CLAUDE_OUTPUT_CONFIG,
            )
            text = "".join(b.text for b in reply.content if b.type == "text")
            return json.loads(text)["answer"]
        except Exception as e:
            log.warning("query failed: %s", e)
            return None

    summary_bits = gzip_bits(summary)
    record = {
        "arm": "claude",
        "model": model,
        "n": len(members),
        "conversation_input_tokens": count.input_tokens,
        "bits_per_item": summary_bits / len(members),
        "summary_gzip_bits": summary_bits,
        "summary_chars": len(summary),
        "summary": summary,  # the artifact itself, for auditing
    }
    record.update(score(*run_queries(ask, member_qs, non_member_qs)))
    return record


def run_claude_full_context(
    members: list[str],
    member_qs: list[str],
    non_member_qs: list[str],
    model: str,
) -> dict:
    """Control that keeps the whole set in context and answers the same queries.

    Any error here is the model's own answering error, not compaction loss. The
    set is sent as one cached block, so it is billed once and read cheaply per
    query rather than re-billed in full."""
    n = len(members)
    listing = "\n".join(members)
    full_bits = 8 * len(listing)

    client = anthropic.Anthropic(timeout=300.0, max_retries=3)
    # The full set, cached once and reused across queries.
    base = [
        {"role": "user", "content": SCAN_PROMPT},
        {"role": "assistant", "content": [
            {"type": "text", "text": listing,
             "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": DONE_PROMPT},
        {"role": "assistant", "content": "OK"},
    ]
    log.info("full-context control: %d items (%d chars) held in context, no "
             "compaction", n, len(listing))

    # No compaction beta here, so the full set stays in context. Prompt caching
    # is GA and needs no beta header.
    def ask(item: str) -> str | None:
        try:
            reply = client.messages.create(
                model=model,
                max_tokens=QUERY_MAX_TOKENS,
                messages=base + [
                    {"role": "user", "content": QUERY_PROMPT.format(item=item)}],
                output_config=CLAUDE_OUTPUT_CONFIG,
            )
            text = "".join(b.text for b in reply.content if b.type == "text")
            return json.loads(text)["answer"]
        except Exception as e:
            log.warning("query failed: %s", e)
            return None

    record = {"arm": "claude-full", "model": model, "n": n,
              "bits_per_item": full_bits / n, "full_context_chars": len(listing)}
    record.update(score(*run_queries(ask, member_qs, non_member_qs)))
    return record


def run_openai(
    members: list[str],
    member_qs: list[str],
    non_member_qs: list[str],
    model: str,
) -> dict:
    """OpenAI standalone compaction, default behavior. The compacted window
    is retained items plus one opaque, encrypted compaction item, so the
    measured size is of the artifact carried forward, not a readable summary.
    https://developers.openai.com/api/docs/guides/compaction
    """
    client = OpenAI(timeout=300.0, max_retries=3)
    log.info("compacting %d items with %s", len(members), model)
    compacted = client.responses.compact(model=model, input=conversation(members))
    window = [
        item.model_dump(exclude_none=True, warnings=False)
        if hasattr(item, "model_dump") else item
        for item in compacted.output
    ]
    serialized = json.dumps(window, sort_keys=True, default=str)
    log.info("compacted: window %d items (%.2f bits/item)",
             len(window), gzip_bits(serialized) / len(members))
    log.info("compacted window:\n%s", json.dumps(window, indent=1, default=str))

    def ask(item: str) -> str | None:
        try:
            response = client.responses.create(
                model=model,
                input=window
                + [{"role": "user", "content": QUERY_PROMPT.format(item=item)}],
                max_output_tokens=QUERY_MAX_TOKENS,
                store=False,
                text=OPENAI_TEXT_FORMAT,
            )
            return json.loads(response.output_text)["answer"]
        except Exception as e:
            log.warning("query failed: %s", e)
            return None

    window_bits = gzip_bits(serialized)
    record = {
        "arm": "openai",
        "model": model,
        "n": len(members),
        "bits_per_item": window_bits / len(members),
        "compacted_window_gzip_bits": window_bits,
        "num_compacted_items": len(window),
        "compacted_window": window,  # the artifact itself, for auditing
    }
    record.update(score(*run_queries(ask, member_qs, non_member_qs)))
    return record


def dump_prompts(args: argparse.Namespace, members, member_qs, non_member_qs):
    """Write the exact endpoint inputs to data/prompts/ for reproducibility."""
    model = args.claude_model if args.arm == "claude" else args.openai_model
    bundle = {
        "arm": args.arm,
        "model": model,
        "dataset": "urls",
        "n": args.n,
        "seed": args.seed,
        "conversation": conversation(members),
        "query_prompt_template": QUERY_PROMPT,
        "member_queries": member_qs,
        "non_member_queries": non_member_qs,
        "answer_schema": ANSWER_SCHEMA,
    }
    if args.arm == "claude":
        bundle["betas"] = [CLAUDE_COMPACTION_BETA]
        bundle["context_management"] = CONTEXT_MANAGEMENT
        bundle["query_request"] = {
            "max_tokens": QUERY_MAX_TOKENS,
            "output_config": CLAUDE_OUTPUT_CONFIG,
            "note": "query continues from the compaction block alone",
        }
    else:
        bundle["query_request"] = {
            "max_output_tokens": QUERY_MAX_TOKENS,
            "store": False,
            "text": OPENAI_TEXT_FORMAT,
        }
    path = args.prompts_dir / f"{args.arm}_n{args.n}_seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)
    log.info("wrote %s", path)


def cmd_run(args: argparse.Namespace) -> None:
    if args.num_queries < 2 or args.num_queries % 2:
        raise SystemExit("--num-queries must be an even number >= 2")
    if args.num_queries // 2 > args.n:
        raise SystemExit("--num-queries // 2 cannot exceed --n")
    members, non_members = url_dataset(
        args.n, args.num_queries, args.seed, args.urls_csv
    )
    member_qs, non_member_qs = make_queries(
        members, non_members, args.num_queries, args.seed + 1
    )
    dump_prompts(args, members, member_qs, non_member_qs)

    if args.arm == "claude" and args.no_compaction:
        record = run_claude_full_context(
            members, member_qs, non_member_qs, args.claude_model
        )
    elif args.arm == "claude":
        record = run_claude(
            members, member_qs, non_member_qs, args.claude_model
        )
    else:
        record = run_openai(
            members, member_qs, non_member_qs, args.openai_model
        )
    record.update(
        dataset="urls",
        seed=args.seed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info(
        "%s done: %.2f bits/item, error=%.3f (FPR=%.3f, FNR=%.3f), failed=%d -> %s",
        record["arm"], record["bits_per_item"], record["error_rate"],
        record["false_positive_rate"], record["false_negative_rate"],
        record["num_failed_queries"], args.out,
    )


def cmd_plot(args: argparse.Namespace) -> None:
    records = []
    if args.results.exists():
        with open(args.results, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    # Skip records missing plotted keys; keep the latest record per
    # configuration so re-runs do not double-plot.
    required = ("arm", "dataset", "n", "seed", "bits_per_item", "error_rate",
                "num_members_queried", "num_non_members_queried")
    valid = [r for r in records if all(k in r for k in required)]
    if len(valid) < len(records):
        log.info("skipped %d malformed record(s)", len(records) - len(valid))
    deduped = {(r["arm"], r["dataset"], r["n"], r["seed"]): r for r in valid}
    if len(deduped) < len(valid):
        log.info("dropped %d duplicate run(s), keeping the latest",
                 len(valid) - len(deduped))
    records = list(deduped.values())

    # OpenAI returns encrypted compaction output whose size is not comparable to
    # a Bloom filter, so plot Claude only.
    records = [r for r in records if r.get("arm") == "claude"]

    # Membership error rate = incorrect answers / total queries (what
    # score() records). Counts both error types, so an all-"no" compactor
    # scores ~0.5 (chance) rather than a misleadingly low FPR. Bloom and
    # the lower bound are given as false-positive rates and err only on
    # non-members, so their error rate is FPR times the non-member
    # fraction of the queries -- the same total-errors/total-queries
    # definition used for the points.
    nm = sum(r["num_non_members_queried"] for r in records)
    tot = sum(r["num_members_queried"] + r["num_non_members_queried"]
              for r in records)
    nonmember_frac = nm / tot if tot else 0.5

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3 (crisp, venue-compliant)
        "ps.fonttype": 42,
    })

    # The x-axis is the total compaction budget in Kbits. Bloom's frontier is
    # a per-item rate, so convert with the (shared) item count n.
    n_items = records[0]["n"] if records else 1
    # Budget is the raw summary size in bits, chars times 8.
    totals = [r["summary_chars"] * 8 / 1000 for r in records]

    fig, ax = plt.subplots(figsize=(6, 4.0))
    xmax = max(totals) * 1.15 if totals else 2.0 * n_items / 1000
    b = np.linspace(0.0, xmax, 400)
    bloom = [bloom_fpr(v * 1000 / n_items) * nonmember_frac for v in b]
    bound = [lower_bound_fpr(v * 1000 / n_items) * nonmember_frac for v in b]

    # Below the Bloom curve is lower error than a Bloom filter (the dashed
    # lower bound is the floor); above 0.5 is worse than a random guess. The
    # compactors sit at the latter boundary.
    ax.fill_between(b, 0, bloom, color="#2ca02c", alpha=0.08, lw=0, zorder=0)
    ax.axhspan(0.5, 1.0, color="#d62728", alpha=0.06, zorder=0)
    ax.text(0.04, 0.94, "Worse Than Random Guess", transform=ax.transAxes,
            fontsize=9, color="#b03030", style="italic")
    ax.text(0.04, 0.06, "Better Than Bloom Filter", transform=ax.transAxes,
            fontsize=9, color="#2e7d32", style="italic")

    ax.plot(b, bloom, "-", color="black", lw=2.2, zorder=3)
    ax.plot(b, bound, "--", color="#2ca02c", lw=1.8, zorder=3)
    ax.axhline(0.5, color="#d62728", linestyle=":", lw=1.5, zorder=2)

    display = {"claude": "Opus 4.8", "openai": "GPT 5.5"}
    markers = {"claude": "x", "openai": "o"}  # one point per run, by shape
    by_arm: dict[str, list] = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in by_arm.items():
        xs = [r["summary_chars"] * 8 / 1000 for r in rs]
        ys = [r["error_rate"] for r in rs]
        ax.plot(xs, ys, marker=markers.get(arm, "o"), linestyle="none",
                markersize=7, color="black", markerfacecolor="none",
                markeredgewidth=1.3, zorder=5)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Compaction Budget (Kbits)")
    ax.set_ylabel("Error Rate")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.grid(True, color="0.9", lw=0.6, zorder=0)

    # Build the legend by hand so the endpoint entries are clean dots rather
    # than errorbar handles with protruding caps.
    handles = [
        Line2D([], [], color="black", lw=2.2, label="Bloom Filter"),
        Line2D([], [], color="#2ca02c", lw=1.8, linestyle="--",
               label="Lower Bound"),
        Line2D([], [], color="#d62728", lw=1.5, linestyle=":",
               label="Random Guess"),
    ]
    handles += [
        Line2D([], [], marker=markers.get(a, "o"), linestyle="none",
               markersize=8, color="black", markerfacecolor="none",
               markeredgewidth=1.3, label=display.get(a, a))
        for a in by_arm
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True,
                    framealpha=0.95, edgecolor="0.8")
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "membership_pareto.pdf", bbox_inches="tight")
    log.info("wrote %s", args.out_dir / "membership_pareto.pdf")

    if records:
        fields = ["arm", "dataset", "model", "n", "seed", "bits_per_item",
                  "error_rate", "false_positive_rate", "false_negative_rate",
                  "num_failed_queries"]
        with open(args.out_dir / "fn_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda r: (r["arm"], r["dataset"])))
        log.info("wrote %s", args.out_dir / "fn_table.csv")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence per-request HTTP logs; our own progress lines are the signal.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one experiment configuration")
    run_p.add_argument("--arm", required=True, choices=["claude", "openai"])
    run_p.add_argument(
        "--n", type=int, default=15000,
        help="recorded items; the Claude arm needs the conversation to "
        "exceed 50,000 tokens, and OpenAI's compaction request must fit "
        "its per-minute token limit")
    run_p.add_argument("--num-queries", type=int, default=200,
                       help="total queries; half members, half non-members")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--no-compaction", action="store_true",
                       help="control: keep the full set in context to isolate "
                       "the model's answering error")
    run_p.add_argument("--claude-model", default=CLAUDE_MODEL)
    run_p.add_argument("--openai-model", default=OPENAI_MODEL)
    run_p.add_argument("--urls-csv", type=Path,
                       default=Path("data/malicious_phish.csv"))
    run_p.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    run_p.add_argument("--out", type=Path, default=Path("results/runs.jsonl"))
    run_p.set_defaults(func=cmd_run)

    plot_p = sub.add_parser("plot", help="write the figure and FN table")
    plot_p.add_argument("--results", type=Path, default=Path("results/runs.jsonl"))
    plot_p.add_argument("--out-dir", type=Path, default=Path("results"))
    plot_p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
