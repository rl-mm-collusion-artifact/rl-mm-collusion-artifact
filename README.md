# Exchange Rules and Tacit Collusion in a Stylized RL Market-Making Game

Code and data for the paper. Independent Q-learning market makers learn tacit
collusion on supra-competitive spreads; we measure which exchange-design levers
(maker rebate, tick size, tie-breaking rule, number of makers) suppress it, and
check the result under a stochastic mid-price and endogenous inventory.

The compiled paper is `collusion/paper.pdf`.

## Install

```
pip install numpy pandas matplotlib
```

Python 3.10+. Building the paper itself also needs the `acmart` LaTeX class.

## Reproduce

Run everything from this directory (the repo root). Sweeps are parallel; set
`--workers` to your core count and pin BLAS threads first:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# tests
python -m pytest collusion/test_collusion.py -q

# sweeps  (-> collusion/results/*.csv)
python -m collusion.sweep            --full   --seeds 20 --out collusion/results/full_sweep20.csv
python -m collusion.sweep            --design --seeds 20 --out collusion/results/design_sweep20.csv
python -m collusion.sweep            --robust --seeds 20 --out collusion/results/robust_sweep20.csv
python -m collusion.sweep_inventory           --seeds 20 --out collusion/results/inventory_sweep.csv

# tables and figures (read the CSVs above)
python collusion/tables.py            # -> collusion/results/tables.tex
python collusion/hardening.py         # -> collusion/results/table_raw.tex
python collusion/inventory_table.py   # -> collusion/results/table_inventory.tex
python -m collusion.figures           # -> collusion/results/figure_*.png

# paper
cd collusion && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Every cell is seeded; results are deterministic given the seed range.

## Layout

```
collusion/
  env.py                  stylized spread-setting game, benchmarks, collusion index
  env_inventory.py        extension: stochastic mid-price + endogenous inventory
  qlearning.py            multi-agent Q-learning; forced-deviation experiment
  qlearning_inventory.py  inventory-aware learner
  sweep.py                parallel condition / design sweeps
  sweep_inventory.py      design sweep in the inventory game
  figures.py tables.py hardening.py inventory_table.py   paper figures and tables
  test_collusion.py       correctness tests
  results/                sweep outputs, generated tables, figures
  paper.tex paper.pdf collusion_refs.bib   the paper
```
