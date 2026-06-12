# context-compaction-experiments

Empirical study for the paper *Context Compaction Theory*: do the
compaction endpoints deployed in production match a Bloom filter on set
membership queries?

The experiment records a set of malicious URLs in a conversation,
compacts the conversation with a production compaction endpoint, asks
membership queries on the compacted output, and compares the measured
size and error rates against the Bloom filter's analytic size-error
frontier. URL blocklists are the canonical Bloom filter workload. Items
are sampled from the [Kaggle Malicious URLs
dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
(see `data/README.md` for download); non-member queries are held-out
URLs from the same dataset, so membership cannot be guessed from a URL's
surface form. Everything lives in `experiment.py`, run with
[uv](https://docs.astral.sh/uv/) (dependencies are declared inline).

## Design

* **Bloom filter**: no runs needed. The frontier `2^(-b ln 2)` at `b`
  bits per item and the lower bound `2^(-b)` appear on the figure
  analytically.
* **`claude`**: server-side compaction (beta `compact-2026-01-12`),
  trigger at its 50,000-token minimum, with custom instructions that ask
  the model to preserve the information needed to answer membership
  queries — a best-effort shot, not the default summarize-to-continue
  behavior. (The default instructions also trigger the model's cyber
  safeguard on bulk content; the explicit membership instructions clear
  it.) The summary's gzip size is the measured size; queries continue
  from the compaction block. Uses `claude-opus-4-8`: the newest model
  (`claude-fable-5`) refuses to compact a bulk URL list under its cyber
  safeguard, so the most recent model that performs the task is used.
* **`openai`**: standalone `responses.compact`. The endpoint exposes no
  instruction control, so it runs as-is. The compacted window (retained
  items plus one opaque compaction item) is the measured artifact.
  Queries resubmit the window plus the question.

The URLs sit in an assistant turn, mirroring how agent state accumulates;
OpenAI's compaction retains user messages verbatim. The conversation
announces that membership queries will follow, so each compactor knows
its workload, just as a Bloom filter does. Answers are constrained to
YES/NO via structured outputs, so no parsing is involved.

## Running

```sh
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
./run_all.sh                  # or: N=10000 SEEDS="42 43" ./run_all.sh
```

Single runs: `uv run experiment.py run --arm claude --n 20000`.

Each run appends one record to `results/runs.jsonl`, including the
compacted artifact itself, and writes the exact endpoint inputs to
`data/prompts/` for reproducibility. `run_all.sh` documents the choice
of N (bounded by the context window, not the dataset) and of a single
seed, and costs roughly $10-15.

## Figure and table

```sh
uv run experiment.py plot
```

Writes `results/membership_pareto.pdf` (membership error rate vs total
compaction budget in Kbits; the Bloom frontier and its lower bound as
curves, each run as one point, marker shape per endpoint) and
`results/fn_table.csv` (per-run bits, error rate, and the false-positive
and false-negative rates behind it). The error rate counts both error
types, so answering NO to everything scores at chance rather than a
misleadingly low false-positive rate.

## Tests and lint

```sh
uv run --with pytest pytest -q
uvx ruff check .
```
