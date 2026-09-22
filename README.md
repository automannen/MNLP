# MNLP

## Data pipeline

All languages/directions/model settings are defined in [`config.yaml`](config.yaml)
(max 6 directions, single pairs work fine too).

```bash
# One-off, run where internet is available (e.g. Snellius login node):
python download_dataset.py --config config.yaml

# Load/align/validate/save the configured directions and build dataloaders:
python main.py --config config.yaml
```

FLORES data is cached under `dataset.data_dir` (default `data/flores/`) so that
compute nodes without internet access can reuse it. Standardized, validated
`{id, src_lang, tgt_lang, source, reference}` records are written as JSONL under
`processed_dir` (default `data/processed/<split>/<src>-<tgt>.jsonl`).
