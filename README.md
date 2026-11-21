<div align="center">

# 🧠 Like a Child's Mind

### *How Collateral Damage in Machine Unlearning Scales Down to Small Language Models*

![Venue](https://img.shields.io/badge/MULVA-2026-6E8B7A?style=flat-square)
![Models](https://img.shields.io/badge/models-SmolLM2%20·%20Qwen2.5-C06C3E?style=flat-square)
![Compute](https://img.shields.io/badge/Colab-A100-2F4858?style=flat-square)
![Status](https://img.shields.io/badge/paper-in%20review-E0A458?style=flat-square)

*Sayanjib Sur · Swastika Ghosh · Surajit Marddnya · Popita Mukherjee*
**Jadavpur University**

</div>

---

## ✦ The idea in one line

> Small language models forget **messily** — erasing one fact damages unrelated knowledge far more
> than it does in large models, because in a compact model those memories share the same weights.

Almost all machine-unlearning methods are built and tuned on 7B+ models. But the models that run on
phones and edge devices are one to two orders of magnitude smaller — and privacy law applies to them
just the same. We ask what unlearning actually **costs** at that scale, and find the cost scales the
wrong way.

## ✦ What we find

<div align="center">

| | **135M** | **360M** | **1.7B** |
|:--|:--:|:--:|:--:|
| General knowledge kept after unlearning *(SmolLM2, NPO)* | **34%** | 44% | **59%** |
| Damage per unit forgotten *(ratio κ, ↓ better)* | **1.09** | 0.83 | **0.57** |

</div>

- **Smaller ⇒ more collateral damage.** At a *matched forgetting level*, a 135M model gives up nearly
  two-thirds of its general knowledge; a 1.7B model, less than half.
- **The mechanism is capacity.** The weights critical for forgetting and for retaining overlap an
  order of magnitude above chance — the two objectives share directions.
- **A cheap fix helps.** Freezing the most retention-critical weights with a **Fisher mask** lifts
  retained knowledge by **+43%** at 135M, at the same forgetting level — most where models are
  smallest and the damage is worst.

## ✦ Repository

```
.
├── notebooks/   Colab notebooks (A100) that produce every result
│   ├── NB1_scaling_benchmark.ipynb         baselines + Fisher diagnostic
│   ├── NB2_fisher_masked_unlearning.ipynb  Fisher-Masked NPO (3-epoch)
│   ├── NB2b_fisher_masked_ep8.ipynb        Fisher-Masked NPO (matched forgetting)
│   └── NB3_final_results.ipynb             final runs + all paper assets
└── docs/        execution plans
```

> 📄 **The paper PDF will be added here once it is finalized.**

## ✦ Reproducing the results

The notebooks run on **Google Colab with an A100 GPU** and checkpoint to Google Drive — the whole
study takes a few A100-hours.

1. **`notebooks/NB1_scaling_benchmark.ipynb`** — memorizes each model, runs the GradDiff/NPO
   baselines at matched forgetting, and saves the Fisher importances. Set `DRIVE_ROOT` in cell 1 to a
   fresh Drive folder.
2. **`notebooks/NB3_final_results.ipynb`** — loads NB1's artifacts, runs Fisher-Masked NPO and the
   ablation, and writes every figure and table (plus `paper_assets/results.json`).

`NB2` / `NB2b` are intermediate experiments kept for reference.

## ✦ Citation

```bibtex
@inproceedings{sur2026childsmind,
  title     = {Like a Child's Mind: How Collateral Damage in Machine Unlearning
               Scales Down to Small Language Models},
  author    = {Sur, Sayanjib and Ghosh, Swastika and Marddnya, Surajit and Mukherjee, Popita},
  booktitle = {MULVA Workshop at ICVGIP},
  year      = {2026}
}
```
