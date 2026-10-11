"""Pre-fetch FLORES-200 data required by config.yaml and cache it locally.

Run this once on a machine with internet access (e.g. the Snellius login node)
before submitting a SLURM job, since compute nodes have no internet access.
Languages already cached under dataset.data_dir are skipped.
"""

import argparse

from dataset_pipeline import get_required_languages, load_config, load_flores_language


def download_all(config):
    dataset_name = config["dataset"]["name"]
    split = config["dataset"]["split"]
    data_dir = config["dataset"]["data_dir"]

    for lang in get_required_languages(config):
        print(f"[flores] {lang} ({split}) ...")
        load_flores_language(lang, split, data_dir, dataset_name)
    print("[flores] done, all required languages are cached locally")


def main():
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 language subsets required by config.yaml"
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    download_all(config)


if __name__ == "__main__":
    main()
