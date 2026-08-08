# NERSC Allocation-Graph Poster Artifact

This repository reproduces the major network-analysis figures from *Understanding the NERSC User Community though Network Analysis* from one cumulative user–repository–year membership table.

The included input is a privacy-safe synthetic table with the columns `repo`, `user_id`, `year`, `office`, and `organization_role`. Replace `data/synthetic_perlmutter_memberships_2020_2026.csv` with another cleaned cumulative table of the same schema, or edit `DATA_PATH` in a notebook.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in any order. Figures are written to `outputs/`.

## Notebooks

1. `01_degree_distribution.ipynb` — degree distribution comparison between NERSC and other networks.
2. `02_rsa_vs_unweighted_ranking.ipynb` — top-100 scientific-role matches in 2025.
3. `03_program_office_composition.ipynb` — office composition and HEP representation.
4. `04_role_composition.ipynb` — role composition by degree and RSA strength.
5. `05_network_metric_methods.ipynb` — compact demonstration of the preliminary table metrics.

No real NERSC user or project records are included.
