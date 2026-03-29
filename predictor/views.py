from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from django.shortcuts import render

from model_generators.classification.train_classifier import (
    FEATURES as ML_FEATURES,
    MODEL_PATH as CLASSIFICATION_MODEL_PATH,
    evaluate_classification_model,
    train_and_save_classification_model,
)
from model_generators.clustering.train_cluster import (
    MAPPING_PATH,
    MODEL_PATH as CLUSTERING_MODEL_PATH,
    SCALER_PATH,
    evaluate_clustering_model,
    train_and_save_clustering_model,
)
from model_generators.regression.train_regression import (
    MODEL_PATH as REGRESSION_MODEL_PATH,
    evaluate_regression_model,
    train_and_save_regression_model,
)
from predictor.data_exploration import data_exploration, dataset_exploration

try:
    import plotly.express as px
except ImportError:
    px = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / 'dummy-data' / 'vehicles_ml_dataset.csv'
RWANDA_GEOJSON_PATH = BASE_DIR / 'dummy-data' / 'rwanda_districts.geojson'


def _ensure_models() -> None:
    if not REGRESSION_MODEL_PATH.exists():
        train_and_save_regression_model()
    if not CLASSIFICATION_MODEL_PATH.exists():
        train_and_save_classification_model()
    if not CLUSTERING_MODEL_PATH.exists() or not SCALER_PATH.exists() or not MAPPING_PATH.exists():
        train_and_save_clustering_model()


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r'[^a-z0-9]+', ' ', str(value).strip().lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


def _feature_centroid(feature: dict) -> Tuple[float, float] | None:
    geom = feature.get('geometry') or {}
    coords = []
    if geom.get('type') == 'Polygon':
        coords = geom.get('coordinates', [])[:1]
    elif geom.get('type') == 'MultiPolygon':
        coords = [poly[0] for poly in geom.get('coordinates', []) if poly]
    if not coords:
        return None
    lons: List[float] = []
    lats: List[float] = []
    for ring in coords:
        for lon, lat in ring:
            lons.append(lon)
            lats.append(lat)
    if not lons or not lats:
        return None
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _build_district_map(df: pd.DataFrame) -> Tuple[str, str, str]:
    district_counts = df.groupby('district').size().reset_index(name='clients')

    if px is None:
        return (
            district_counts.to_html(
            classes='table table-bordered table-striped table-sm',
            index=False,
            ),
            '',
            district_counts.sort_values('district').to_html(
                classes='table table-bordered table-striped table-sm',
                index=False,
            ),
        )

    if RWANDA_GEOJSON_PATH.exists():
        with RWANDA_GEOJSON_PATH.open('r', encoding='utf-8') as f:
            geojson = json.load(f)

        features = geojson.get('features', [])
        geo_names = [feature['properties'].get('shapeName') for feature in features]
        norm_to_geo: Dict[str, str] = {
            _normalize_name(name): name for name in geo_names if name
        }

        district_counts['district_normalized'] = district_counts['district'].map(_normalize_name)
        district_counts['district_mapped'] = district_counts['district_normalized'].map(norm_to_geo)
        unmatched = district_counts[district_counts['district_mapped'].isna()]

        base = pd.DataFrame({'district': sorted(geo_names)})
        mapped = district_counts.dropna(subset=['district_mapped']).copy()
        mapped = mapped[['district_mapped', 'clients']].rename(columns={'district_mapped': 'district'})
        district_counts = base.merge(mapped, on='district', how='left').fillna(0)
        district_counts['clients'] = district_counts['clients'].astype(int)

        fig = px.choropleth(
            district_counts,
            geojson=geojson,
            locations='district',
            featureidkey='properties.shapeName',
            color='clients',
            color_continuous_scale='Blues',
            title='Vehicle Clients by Rwanda District',
            hover_data={'district': True, 'clients': True},
        )
        fig.update_traces(
            marker_line_width=1.3,
            marker_line_color='#111827',
            marker_opacity=0.9,
            hovertemplate='<b>%{location}</b><br>Clients: %{z}<extra></extra>',
        )
        count_map = {
            row['district']: int(row['clients'])
            for _, row in district_counts.iterrows()
        }
        label_lons: List[float] = []
        label_lats: List[float] = []
        label_text: List[str] = []
        for feature in features:
            name = feature['properties'].get('shapeName')
            centroid = _feature_centroid(feature)
            if name and centroid:
                label_lons.append(centroid[0])
                label_lats.append(centroid[1])
                label_text.append(f"{name}<br>{count_map.get(name, 0)}")
        if label_lons and label_lats:
            fig.add_scattergeo(
                lon=label_lons,
                lat=label_lats,
                text=label_text,
                mode='text',
                textfont=dict(size=9, color='#23395d'),
                showlegend=False,
                hoverinfo='skip',
            )
        # Focus on Rwanda only; hide base map features so only district boundaries show.
        centroid = None
        if label_lons and label_lats:
            centroid = {
                'lon': sum(label_lons) / len(label_lons),
                'lat': sum(label_lats) / len(label_lats),
            }

        fig.update_geos(
            fitbounds='locations',
            visible=False,
            showcountries=False,
            showcoastlines=False,
            showframe=False,
            showland=False,
            showlakes=False,
            showrivers=False,
            showsubunits=False,
            projection_type='mercator',
            center=centroid,
            projection_scale=6,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520, paper_bgcolor='white')

        unmatched_html = ''
        if not unmatched.empty:
            unmatched_html = unmatched[['district', 'clients']].to_html(
                classes='table table-bordered table-striped table-sm',
                index=False,
            )
    else:
        fig = px.bar(
            district_counts,
            x='district',
            y='clients',
            title='Vehicle Clients by District (Fallback Chart)',
            color='clients',
            color_continuous_scale='Blues',
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=430)
        unmatched_html = ''

    counts_html = district_counts.sort_values('district').to_html(
        classes='table table-bordered table-striped table-sm',
        index=False,
    )
    return fig.to_html(full_html=False), unmatched_html, counts_html


_ensure_models()
regression_model = joblib.load(REGRESSION_MODEL_PATH)
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
clustering_scaler = joblib.load(SCALER_PATH)
cluster_mapping = joblib.load(MAPPING_PATH)


def data_exploration_view(request):
    df = pd.read_csv(DATASET_PATH)
    map_result = _build_district_map(df)
    if len(map_result) == 2:
        district_map, unmatched_districts = map_result
        district_counts = df.groupby('district').size().reset_index(name='clients').to_html(
            classes='table table-bordered table-striped table-sm',
            index=False,
        )
    else:
        district_map, unmatched_districts, district_counts = map_result
    context = {
        'data_exploration': data_exploration(df),
        'dataset_exploration': dataset_exploration(df),
        'district_map': district_map,
        'unmatched_districts': unmatched_districts,
        'district_counts': district_counts,
    }
    return render(request, 'predictor/index.html', context)


def regression_analysis(request):
    context = {'evaluations': evaluate_regression_model()}
    if request.method == 'POST':
        year = int(request.POST['year'])
        km = float(request.POST['km'])
        seats = int(request.POST['seats'])
        income = float(request.POST['income'])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context['price'] = prediction
    return render(request, 'predictor/regression_analysis.html', context)


def classification_analysis(request):
    context = {'evaluations': evaluate_classification_model()}
    if request.method == 'POST':
        year = int(request.POST['year'])
        km = float(request.POST['km'])
        seats = int(request.POST['seats'])
        income = float(request.POST['income'])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context['prediction'] = prediction
    return render(request, 'predictor/classification_analysis.html', context)


def clustering_analysis(request):
    context = {'evaluations': evaluate_clustering_model()}
    if request.method == 'POST':
        try:
            year = int(request.POST['year'])
            km = float(request.POST['km'])
            seats = int(request.POST['seats'])
            income = float(request.POST['income'])

            reg_input = pd.DataFrame([[year, km, seats, income]], columns=ML_FEATURES)
            predicted_price = float(regression_model.predict(reg_input)[0])

            cluster_input = clustering_scaler.transform([[income, predicted_price]])
            cluster_id = int(clustering_model.predict(cluster_input)[0])

            context.update(
                {
                    'prediction': cluster_mapping.get(cluster_id, 'Unknown'),
                    'price': predicted_price,
                }
            )
        except Exception as exc:
            context['error'] = str(exc)
    return render(request, 'predictor/clustering_analysis.html', context)
