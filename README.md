# Django Machine Learning Lab - Vehicle Analytics System

This project implements:
- Regression: predict `selling_price`
- Classification: predict `income_level`
- Clustering: segment clients (`Economy`, `Standard`, `Premium`)
- EDA dashboard with Rwanda district client visualization

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python manage.py runserver
```

Open: `http://127.0.0.1:8000`

## Notes

- Dataset: `dummy-data/vehicles_ml_dataset.csv`
- Rwanda district geojson: `dummy-data/rwanda_districts.geojson`
- Trained models are saved under `model_generators/*/*.pkl`
- Clustering page includes Silhouette Score and Coefficient of Variation.
