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


To run the evaluation pipeline the following commands can be used:
To get the metric scores: Chrf++, COMET, and TER run the following command:

```bash
python run_evaluation.py --checkpoints <checkpoints> --out_csv results/__.csv --output <output_dir>
```
An example can be found in the run_evaluation_devtest.job. 
To run with the devtest add: --config config_devtest.yaml to the command above.
The script computes the scores over all the outputs in the specified <output_dir> so make sure to specify the correct directory or if you have a new batch of experiments, make sure to create a new directory for the outputs.
The script will create a CSV file and write the scores for each checkpoint to the CSV file. It is not an append but a write so if you have a previous CSV file with the same name, it will be overwritten.

To get the significance scores, run the following command:

```bash
python evaluate.py --compare <checkpoint1> <checkpoint2> --bootstrap_samples <num_samples>
```

The script will compute the significance scores between the two checkpoints specified. It will store the results in a CSV file named "compare_<checkpoint1>_<checkpoint2>.csv" in the results directory. Again the script will make a csv and write to it so make sure that the results do not get overwritten if you have a previous CSV file with the same name. See the compare_devtest.job for an example of how to run the script with the devtest set. 