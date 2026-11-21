# Like a Child's Mind — MULVA 2026 submission

LaTeX source for *"Like a Child's Mind: How Collateral Damage in Machine
Unlearning Scales Down to Small Language Models."*

## Files
| File | Purpose |
| :-- | :-- |
| `main.tex` | The paper (ACM `sigconf`). All figures are inline TikZ/pgfplots — there are **no external image files**, so the source is fully self-contained and reproducible. |
| `references.bib` | Bibliography database (16 entries). |
| `main.pdf` | Pre-compiled output (7 pages). |

> **Note on the ACM class.** `acmart.cls` and `ACM-Reference-Format.bst` are **not**
> bundled on purpose. They ship with Overleaf and every TeX Live install, and
> bundling an older copy caused a `hyperxmp`/`hyperref` load-order error against
> newer distributions. Use your distribution's current `acmart` (if an old version
> is installed locally, run `tlmgr update acmart hyperxmp`).

## Build

Overleaf: upload `main.tex` + `references.bib` and compile (acmart is built in).

Local, easiest (handles the bibtex passes automatically):

```bash
tectonic main.tex
```

Or with a standard TeX Live install:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Notes
- **Author emails** are intentionally left blank in `main.tex` — fill the
  `\email{}` fields before submission.
- Each figure carries an `\Description{}` for accessibility (satisfies acmart's
  alt-text check).
- The bibliography compiles cleanly. BibTeX may still print a few cosmetic
  `empty address` / `page numbers missing` notes for conference entries; these do
  not affect the output, and several venues (ICLR, COLM) have no page numbers to
  supply. Fill them only if the venue's style requires it.
- **Before submitting,** double-check each reference's venue/year in
  `references.bib`, and switch `\documentclass[sigconf,nonacm]{acmart}` to the
  venue's required options (drop `nonacm`, add the ICVGIP/MULVA copyright block)
  for the camera-ready.
