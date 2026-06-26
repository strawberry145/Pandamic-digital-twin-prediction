# Pandemic Digital Twin 

This repository now includes `pandemic_digital_twin.py`, a prototype pipeline for pandemic risk and case prediction using `public_health_surveillance_dataset.csv`.

## What it does
- loads and preprocesses the health surveillance dataset
- trains individual-level models for:
  - outbreak status classification
  - infection risk level classification
  - daily new cases regression
- trains region-level forecast models for:
  - next-day case forecasts
  - next-week case forecasts

## Run
```bash
python pandemic_digital_twin.py --csv public_health_surveillance_dataset.csv
```

## Requirements
- Python 3.10+
- pandas
- scikit-learn

If needed, install dependencies with:
```bash
pip install -r requirements.txt
```
