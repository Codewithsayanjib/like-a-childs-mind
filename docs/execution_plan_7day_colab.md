# MULVA 2026 — Condensed 7-Day Execution Plan (Colab A100 40GB)

**Scope:** Trim the 11-day / 3-scale / 3-baseline blueprint into a **single-GPU (Colab A100 40GB)** study that fits in **≤ 2 notebooks** and **7 days of execution**, while keeping the paper's two headline contributions intact.

---

## 1. What the full plan does vs. what we keep

| Dimension | Full blueprint | Condensed 7-day version | Why |
| :--- | :--- | :--- | :--- |
| **Model scales** | 256M, 500M, 2.2B | **256M + 500M** (2.2B = optional stretch via LoRA) | 2 points still give a scaling trend; 2.2B full-finetune does not fit 40GB cleanly and eats the compute budget. |
| **Baselines** | GradDiff, NPO, RMU | **NPO + GradDiff** (drop RMU) | NPO is the strong SOTA baseline; GradDiff is the cheap reference. RMU adds runs without changing the story. |
| **Regimes** | Text-Only + MM-Naive | **Text-Only + MM-Naive** (kept) | This contrast *is* the cross-modal-bleed finding — the core narrative. |
| **Forget splits** | forget05 + forget10 | **forget10 only** | One split is enough to prove the effect; halves every run count. |
| **Mechanistic Fisher** | Full Jaccard heatmaps across all layers ×3 scales | **One lightweight Fisher-overlap figure** (256M vs 500M) | Keeps the mechanistic hook; a single figure instead of a full sweep. |
| **Proposed method** | CM-FMU (Fisher mask + align loss) | **CM-FMU kept in full** | This is the contribution — do not cut it. |
| **Ablations** | α, λ_align, module-scope (3 sweeps) | **α sweep + λ_align sweep on 500M only** | Two 1-D sweeps give Table 2 / Figure 4; drop the module-scope grid. |
| **Tuning** | Full-param or LoRA | **LoRA (r=16) everywhere** | Fits 40GB, checkpoints are tiny (~50MB), far faster to train and swap. |

**Net result:** the paper still tells the same 3-beat story — (1) cross-modal resurrection exists and worsens as models shrink, (2) we localize it with Fisher overlap, (3) CM-FMU fixes it without utility collapse — just with fewer confirmatory cells.

---

## 2. Notebook layout (2 notebooks)

### Notebook A — `01_data_diagnose.ipynb` (Days 1–3)
Setup, base memorization, baselines, evaluation harness, Fisher diagnostic. Produces **Table 1** + **Figure 3** data.

1. **Env & data** — install `transformers`, `peft`, `trl`, `bitsandbytes`; load `CaptionEmporium/CLEAR` (`forget10` / `retain90` / `real_authors` / `world_facts`); 15-line PIL OCR-probe generator.
2. **Eval harness** — `evaluate()` → `FE_txt`, `FE_mm`, `CMRR`, `MU` as JSON. Validate on off-the-shelf SmolVLM-256M (< 2 min).
3. **Base memorization** — LoRA-finetune 256M + 500M on CLEAR (text+image) to > 80% ROUGE-L on forget split. Save adapters `base_256M`, `base_500M`.
4. **Baselines** — NPO + GradDiff × {Text-Only, MM-Naive} × {256M, 500M} = **8 runs** (~4 min each). Save adapters.
5. **Fisher diagnostic** — accumulate diagonal Fisher over 1 pass on `forget_txt`, `forget_mm`, `retain_mm`; compute layer-wise Jaccard overlap → save tensors + one heatmap.

Checkpoint everything to Google Drive at the end of each section (Colab sessions die).

### Notebook B — `02_cmfmu_ablate_plots.ipynb` (Days 4–6)
CM-FMU method, ablations, and final figures. Produces **Table 2** + **Figures 2, 4, 5**.

6. **CM-FMU trainer** — gradient-mask hook (`grad * M` from Notebook A's Fisher mask) + `L_align` at middle SLM layer. Run on 256M + 500M on forget10.
7. **Ablations (500M only)** — α ∈ {0, 15, 30, 45, 60}% and λ_align ∈ {0, 0.1, 0.5, 1.0}.
8. **Plots** — Figure 2 (Utility↑ vs MM-Forget↓ Pareto), Figure 4 (ablation curves), Figure 5 (qualitative face/OCR resurrection vs CM-FMU). Reload Notebook A's saved JSON/tensors.

---

## 3. Compute budget check (A100 40GB)

| Step | Runs | ~min/run | A100 hours |
| :--- | :---: | :---: | :---: |
| Base memorization (LoRA) | 2 | ~25 | ~0.8 |
| Baselines | 8 | ~4 | ~0.5 |
| Eval all checkpoints | ~12 | ~2 | ~0.4 |
| Fisher accumulation | 2 | ~15 | ~0.5 |
| CM-FMU | 2 | ~8 | ~0.3 |
| Ablations | 9 | ~6 | ~0.9 |
| **Total** | | | **~3.5–4 A100 hrs** |

Comfortably inside Colab session limits. 2.2B as a stretch adds ~1.5 hrs if time remains on Day 6.

**40GB fit:** SmolVLM-500M + LoRA + bf16 + batch 8 + grad-accum → well under 40GB. 256M is trivial. Freeze the vision encoder (already in the method); only connector + SLM layers get adapters.

---

## 4. Revised 7-day schedule

| Day | Focus | Deliverable |
| :---: | :--- | :--- |
| **1** | Notebook A §1–2: data loaders, OCR probe, eval harness | Eval outputs JSON metrics on base 256M |
| **2** | Notebook A §3–4: base memorization + all 8 baselines | Saved adapters, >80% forget ROUGE-L |
| **3** | Notebook A §5: eval all baselines + Fisher diagnostic | **Table 1** raw numbers + **Figure 3** |
| **4** | Notebook B §6: implement + run CM-FMU | CM-FMU checkpoints + metrics |
| **5** | Notebook B §7: α and λ_align ablations | **Table 2** / **Figure 4** data |
| **6** | Notebook B §8: generate all plots + qualitative examples | **Figures 2, 4, 5** as PDFs |
| **7** | Buffer: 2.2B stretch run *or* re-runs, lock all tables/figures | Frozen result assets for writing |

Writing (Sections, abstract, LaTeX formatting) happens **after** these 7 execution days, off-GPU.

---

## 5. Minimum viable paper assets (after trimming)

- **Table 1** — NPO/GradDiff (Text-Only vs MM-Naive) vs CM-FMU on 256M + 500M, forget10 (`FE_txt`, `FE_mm`, `CMRR`, `MU`).
- **Figure 2** — Utility vs MM-Forget-Efficacy Pareto (2 scales, rescue by CM-FMU).
- **Figure 3** — Fisher Jaccard overlap (Forget-vs-Retain shrinking models; Text-vs-Vision across layers).
- **Figure 4 + Table 2** — α and λ_align ablations on 500M.
- **Figure 5** — Qualitative: face/OCR image resurrecting bios after Text-NPO vs clean erase by CM-FMU.

This is a complete, defensible short-paper story; the 2.2B column and RMU row can be added as "additional scales/baselines" if the stretch run and writing time allow.

---

## 6. Practical Colab notes

- **Persist to Drive** after every section — Colab disconnects lose local state; LoRA adapters are small so saving is cheap.
- **Keep `ref_model` for NPO** by loading a frozen copy of the base adapter (not a second 500M full model) to save VRAM.
- **Fisher mask reuse** — compute once in Notebook A, save the boolean mask tensors, load in Notebook B so the method and diagnostic stay consistent.
- **Determinism** — fix seeds; log every run's config to the JSON so tables are reproducible.
- **Stretch gate** — only attempt 2.2B (LoRA, batch 2, grad-accum 8) on Day 7 if Days 1–6 are fully green.
