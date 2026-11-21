# Like a Child's Mind

**How Collateral Damage in Machine Unlearning Scales Down to Small Language Models**

Sayanjib Sur, Swastika Ghosh, Surajit Marddnya, Popita Mukherjee · Jadavpur University
Submitted to the MULVA 2026 Workshop (ICVGIP 2026).

---

## What this is

Almost all machine-unlearning methods are built and tuned on 7B+ models, yet the models that
run on phones and edge devices are far smaller. This project asks what unlearning *costs* at that
scale. Across two model families and three sizes each (**SmolLM2** 135M/360M/1.7B and
**Qwen2.5** 0.5B/1.5B/3B), evaluated at a **matched forgetting level**, we find that the
collateral damage to unrelated general knowledge grows sharply as models shrink — and that a
lightweight **Fisher-masked** intervention, which freezes the weights most critical for retention,
measurably reduces this damage, most at the smallest scale.

## Repository layout

```
.
├── paper/          LaTeX source (ACM sigconf) + compiled PDF
│   ├── main.tex
│   ├── references.bib
│   ├── main.pdf
│   └── README.md   build instructions
├── notebooks/      Colab notebooks (run on an A100)
│   ├── NB1_scaling_benchmark.ipynb        baselines + Fisher diagnostic
│   ├── NB2_fisher_masked_unlearning.ipynb FM-NPO (3-epoch)
│   ├── NB2b_fisher_masked_ep8.ipynb       FM-NPO (matched forgetting)
│   └── NB3_final_results.ipynb            final runs + all paper assets
└── docs/           execution plans
```

## Reproducing the results

The notebooks are written for **Google Colab with an A100 GPU** and checkpoint to Google Drive.

1. Run `notebooks/NB1_scaling_benchmark.ipynb` first — it memorizes each model, runs the
   GradDiff/NPO baselines at matched forgetting, and saves the Fisher importances. Set `DRIVE_ROOT`
   in cell 1 to a fresh Drive folder.
2. Run `notebooks/NB3_final_results.ipynb` — it loads NB1's artifacts, runs Fisher-Masked NPO and
   the ablation, and writes every figure and table (plus `paper_assets/results.json`).

`NB2`/`NB2b` are intermediate experiments kept for reference. The whole study runs in a few
A100-hours.

## Building the paper

```bash
cd paper
tectonic main.tex      # or: pdflatex main; bibtex main; pdflatex main; pdflatex main
```

`acmart.cls` and `ACM-Reference-Format.bst` ship with Overleaf and TeX Live, so they are not
vendored here. See `paper/README.md` for details.

## Citation

```bibtex
@inproceedings{sur2026childsmind,
  title     = {Like a Child's Mind: How Collateral Damage in Machine Unlearning Scales Down to Small Language Models},
  author    = {Sur, Sayanjib and Ghosh, Swastika and Marddnya, Surajit and Mukherjee, Popita},
  booktitle = {MULVA Workshop at ICVGIP},
  year      = {2026}
}
```
