# GitHub submission checklist

## Repository contents

The Git repository contains:

- source code for modules 1 through 6;
- the current LoRA weight and cryptographic provenance record;
- the clean Colab training notebook;
- dependency declarations, tests, documentation, and CI.

The repository intentionally excludes virtual environments, generated outputs,
prepared training archives, legacy model weights, organizer-provided datasets,
color tables, submission templates, and local credentials. Authorized copies
of those inputs must be supplied locally under `reference_data/`.

## Final verification

Before tagging a release:

1. Run all unit tests:

   ```bash
   python -m unittest discover -s modules -p "test_module_*.py" -v
   ```

2. Run the 45-scene evaluation with the release source state:

   ```bash
   python -m modules.module_06_demo_evaluation.src.evaluator \
     --output-dir outputs/release_evaluation
   ```

3. Confirm `batch_summary.json` reports `45` succeeded and `0` failed.

4. Review representative Raw, themed, SDL preview, and SDL control images.

5. Export the official workbook:

   ```bash
   python -m modules.module_06_demo_evaluation.src.exporter \
     --results outputs/release_evaluation/batch_results.jsonl \
     --output outputs/release_evaluation/测试集提交结果.xlsx
   ```

6. Open the workbook and verify all 45 rows, embedded images, English prompts,
   Chinese scenes, and ordering.

## Release assets

Attach these generated files to the GitHub release rather than committing them
to Git history:

- `测试集提交结果.xlsx`
- `batch_results.jsonl`
- `batch_summary.json`
- a curated result archive or representative images;
- the v4 training dataset ZIP only when redistribution is authorized.

Never attach API keys, `.env` files, raw organizer training images, or legacy
intermediate archives.
