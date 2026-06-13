# Context Compaction Theory (Empirical Study)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

To run the experiments, you will need a key to OpenAI and a key to Anthropic. 

```sh
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

You can run the experiments using the following script:

```
run_all.sh
```

Take a look at its commands for more information.

To generate figures using the data from the experiments, you can use

```sh
uv run experiment.py plot
```

To run tests, you can use 

```sh
uv run --with pytest pytest -q
```

Finally, to run a lint check, just use

```
uvx ruff check .
```
