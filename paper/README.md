# Paper build

The manuscript source is `main.tex`. Numerical tables are generated only from the CSV files in
`results/confirmatory`, which are produced by `scripts/aggregate_results.py` from recorded
checkpoint and evaluation metadata. `scripts/build_paper_results.py` replaces
`generated/results.tex`.

```bash
python3 scripts/aggregate_results.py
python3 scripts/plot_results.py
python3 scripts/build_paper_results.py
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The current environment may not have a TeX distribution installed. In that case, keep the
LaTeX source and generated tables/figures in Git and do not claim that a PDF was produced.
