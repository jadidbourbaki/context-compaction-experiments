# Data

The experiment uses the [Kaggle Malicious URLs
dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset).
Download it and place `malicious_phish.csv` (a `url,type` CSV, ~651K
rows) in this directory:

```sh
uv run --with kagglehub python -c "import kagglehub, shutil, os; \
p = kagglehub.dataset_download('sid321axn/malicious-urls-dataset'); \
shutil.copy(os.path.join(p, 'malicious_phish.csv'), 'data/')"
```

`data/malicious_phish.csv` is the default `--urls-csv` path and is
gitignored. `data/prompts/` holds the exact endpoint inputs written by
each run.
