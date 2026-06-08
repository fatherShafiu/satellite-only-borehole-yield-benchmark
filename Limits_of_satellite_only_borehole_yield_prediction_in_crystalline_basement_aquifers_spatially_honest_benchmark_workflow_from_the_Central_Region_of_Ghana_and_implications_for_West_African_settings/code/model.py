import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.impute import KNNImputer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import folium
import joblib, warnings, json
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.contour import ContourSet
from folium.plugins import HeatMap, Fullscreen, MeasureControl
from matplotlib.patches import Polygon as mplPolygon
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from xgboost import XGBRegressor
from scipy.stats import rankdata, spearmanr
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    HAS_GP = True
except ImportError:
    HAS_GP = False
try:
    from sklearn.linear_model import QuantileRegressor
    HAS_QUANTILE = True
except ImportError:
    HAS_QUANTILE = False

warnings.filterwarnings("ignore")

# --- Configuration ---
RANDOM_STATE = 42
N_GEOLOGY_CLUSTERS = 3
RUN_PROFILES = {
    'publication': {
        'temporal_lags': 3,
        'external_buffer_deg': 0.6
    },
    'paper': {
        'temporal_lags': 0,
        'external_buffer_deg': 0.6
    },
    'manuscript': {
        'temporal_lags': 0,
        'external_buffer_deg': 0.6
    },
    'ranked': {
        'temporal_lags': 0,
        'external_buffer_deg': 0.6
    },
    'optimized': {
        'temporal_lags': 0,
        'external_buffer_deg': 0.6
    },
    'optimized_ssl': {
        'temporal_lags': 0,
        'external_buffer_deg': 0.6
    }
}
DATA_SOURCES = {}


def set_global_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


set_global_seed()


def file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_file_metadata(path):
    if not os.path.exists(path):
        return {'path': path, 'exists': False}
    stat = os.stat(path)
    return {
        'path': path,
        'exists': True,
        'size_bytes': stat.st_size,
        'modified_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        'sha256': file_sha256(path)
    }

# High-Visibility Color Scheme
POTENTIAL_BREAKS = [
    (0.0, 0.5, '#FF0000'),      # Bright Red (Very Low)
    (0.5, 1.5, '#FFA500'),      # Orange (Moderate)
    (1.5, 2.0, '#00FF00'),      # Bright Green (High)
    (2.0, 2.5, '#0000FF')       # Bright Blue (Very High)
]

# --- Physics-Informed Loss Function ---
class PhysicsInformedLoss:
    def __init__(self, elevation, recharge_potential, alpha=0.1, beta=0.05):
        self.elevation = torch.tensor(elevation.to_numpy() if isinstance(elevation, pd.Series) else elevation, 
                                    dtype=torch.float32)
        self.recharge_potential = torch.tensor(recharge_potential.to_numpy() if isinstance(recharge_potential, pd.Series) else recharge_potential, 
                                             dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_true, y_pred):
        y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else y_true
        y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else y_pred
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        # Physical constraint: yield should generally decrease with elevation
        sorted_idx = torch.argsort(self.elevation)
        elevation_penalty = torch.mean(F.relu(y_pred[sorted_idx].diff()))

        # Additional constraint: yield should not exceed recharge potential
        recharge_penalty = torch.mean(F.relu(y_pred - self.recharge_potential))

        # Base MSE loss
        mse_loss = F.mse_loss(y_pred, y_true)

        return mse_loss + self.alpha * elevation_penalty + self.beta * recharge_penalty

# --- Custom Stacked Model Class ---
class StackedModel:
    def __init__(self, model1, model2, meta_model):
        self.model1 = model1
        self.model2 = model2
        self.meta_model = meta_model
        self.final_estimator_ = meta_model

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        stacked = np.column_stack([pred1, pred2])
        return self.meta_model.predict(stacked)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def build_ssl_latent_features(
        boreholes_df,
        grid_df,
        feature_cols,
        latent_dim=8,
        hidden_dim=32,
        epochs=180,
        lr=1e-3,
        mask_prob=0.15):
    """Train denoising autoencoder on boreholes+grid predictors and return latent features."""
    usable_cols = [
        c for c in feature_cols
        if c in boreholes_df.columns and c in grid_df.columns
    ]
    if len(usable_cols) < 4:
        return None, None, []

    b = boreholes_df[usable_cols].apply(pd.to_numeric, errors='coerce')
    g = grid_df[usable_cols].apply(pd.to_numeric, errors='coerce')
    combined = pd.concat([b, g], axis=0, ignore_index=True)

    col_medians = combined.median(numeric_only=True)
    combined = combined.fillna(col_medians)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(combined.to_numpy(dtype=np.float32))

    n_boreholes = len(boreholes_df)
    X_tensor = torch.tensor(X_all, dtype=torch.float32)

    set_global_seed(RANDOM_STATE)
    model = DenoisingAutoencoder(
        input_dim=X_all.shape[1],
        latent_dim=min(latent_dim, max(4, X_all.shape[1] // 2)),
        hidden_dim=max(hidden_dim, 16)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mask = (torch.rand_like(X_tensor) > mask_prob).float()
        X_noisy = X_tensor * mask
        X_hat, _ = model(X_noisy)
        loss = F.mse_loss(X_hat, X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, Z_all = model(X_tensor)
    Z_np = Z_all.detach().cpu().numpy()

    latent_cols = [f'ssl_latent_{i}' for i in range(Z_np.shape[1])]
    z_b = Z_np[:n_boreholes, :]
    z_g = Z_np[n_boreholes:, :]

    return z_b, z_g, latent_cols

# --- Helper Functions ---
def load_shapefile(shapefile_path):
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs is None:
            raise ValueError("Shapefile CRS is undefined")
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        raise

def get_shapefile_bounds(gdf):
    bounds = gdf.total_bounds
    return {
        'min_lon': bounds[0],
        'min_lat': bounds[1],
        'max_lon': bounds[2],
        'max_lat': bounds[3]
    }

def parse_geo(geo_str):
    try:
        geo_dict = json.loads(geo_str.replace("'", "\""))
        return pd.Series({'lon': geo_dict['coordinates'][0], 'lat': geo_dict['coordinates'][1]})
    except (ValueError, KeyError, TypeError):
        return pd.Series({'lon': np.nan, 'lat': np.nan})

def safe_save(df, filename):
    try:
        df.to_csv(filename, index=False)
        print(f"Saved to: {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

def impute_nans(df, feature, dataset):
    known = dataset[['lon', 'lat', feature]].dropna()
    if len(known) == 0:
        return df
    tree = cKDTree(known[['lon', 'lat']])
    nan_indices = df[df[feature].isna()].index
    if not nan_indices.empty:
        distances, indices = tree.query(df.loc[nan_indices, ['lon', 'lat']])
        df.loc[nan_indices, feature] = known[feature].values[indices]
    return df

def filter_points_in_polygon(df, gdf):
    if df is None or df.empty:
        return df

    central_polygon = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

    working = df.copy()
    working['lon'] = pd.to_numeric(working['lon'], errors='coerce')
    working['lat'] = pd.to_numeric(working['lat'], errors='coerce')
    working = working.dropna(subset=['lon', 'lat'])

    bbox_mask = (
        (working['lon'] >= min_lon) & (working['lon'] <= max_lon) &
        (working['lat'] >= min_lat) & (working['lat'] <= max_lat)
    )
    working = working.loc[bbox_mask]
    if working.empty:
        return working

    points = gpd.GeoSeries(gpd.points_from_xy(working['lon'], working['lat']), crs='EPSG:4326')
    spatial_mask = points.intersects(central_polygon)
    return working.loc[spatial_mask.values]


def filter_external_datasets_to_region(datasets, gdf, buffer_deg=0.6):
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    min_lon -= buffer_deg
    min_lat -= buffer_deg
    max_lon += buffer_deg
    max_lat += buffer_deg

    filtered = []
    for dataset in datasets:
        if dataset is None or dataset.empty:
            filtered.append(dataset)
            continue

        working = dataset.copy()
        working['lon'] = pd.to_numeric(working['lon'], errors='coerce')
        working['lat'] = pd.to_numeric(working['lat'], errors='coerce')
        working = working.dropna(subset=['lon', 'lat'])

        mask = (
            (working['lon'] >= min_lon) & (working['lon'] <= max_lon) &
            (working['lat'] >= min_lat) & (working['lat'] <= max_lat)
        )
        clipped = working.loc[mask]

        # Keep original data if clipping removes too many support points.
        filtered.append(clipped if len(clipped) >= 20 else working)

    return filtered

def create_geological_proxies(df):
    geo_features = df[['grace_anomaly', 'elevation', 'slope']].copy()
    if geo_features.isna().any().any():
        imputer = KNNImputer(n_neighbors=3)
        geo_features = pd.DataFrame(imputer.fit_transform(geo_features),
                                    columns=geo_features.columns,
                                    index=geo_features.index)
    geo_features = (geo_features - geo_features.mean()) / geo_features.std()

    coords = df[['lon', 'lat']].values
    spatial_clusters = DBSCAN(eps=0.1, min_samples=3).fit_predict(coords)
    df['spatial_group'] = spatial_clusters

    kmeans = KMeans(n_clusters=N_GEOLOGY_CLUSTERS, random_state=RANDOM_STATE)
    df['geo_cluster'] = kmeans.fit_predict(geo_features)
    distances = kmeans.transform(geo_features)
    for i in range(N_GEOLOGY_CLUSTERS):
        df[f'geo_dist_{i}'] = distances[:,i]
    df['voltaian_proxy'] = ((df['geo_dist_0'] < 1) & (df['elevation'] > df['elevation'].median())).astype(int)
    df['birimian_proxy'] = ((df['geo_dist_1'] < 1) & (df['slope'] > 3)).astype(int)
    df['tarkwaian_proxy'] = ((df['geo_dist_2'] < 1) & (df['slope'] > 5)).astype(int)
    df['high_terrain'] = (df['elevation'] > df['elevation'].median()).astype(int)
    df['steep_slope'] = (df['slope'] > 5).astype(int)
    return df

def enhanced_feature_engineering(df, temporal_lags=1):
    required_cols = ['grace_anomaly', 'precipitation', 'soil_moisture', 'elevation', 'slope', 'geo_cluster']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for feature engineering: {col}")

    # Non-linear feature selection inspired by Gamma Test (simplified variance ranking)
    y = df['yield'] if 'yield' in df.columns else np.zeros(len(df))
    feature_scores = {}
    for col in required_cols:
        residuals = y - np.poly1d(np.polyfit(df[col], y, 1))(df[col])
        feature_scores[col] = np.var(residuals)
    selected_features = sorted(feature_scores, key=feature_scores.get)[:5]  # Top 5 features

    # SBGI: Satellite-Based Groundwater Index
    df['sbgi'] = ((df['grace_anomaly'] + df['precipitation']) *
                  np.sqrt(df['soil_moisture'])) / (df['elevation'] + 1e-6)
    df['sbgi_geology'] = df['sbgi'] * (0.8 + 0.4*df['geo_cluster'])
    # Backward-compatible aliases for older analyses
    df['groundwater_index'] = df['sbgi']
    df['gwi_geology'] = df['sbgi_geology']
    df['topo_wetness'] = np.log(np.maximum(df['precipitation'], 1e-6) / (df['slope'] + 0.1))
    df['elev_precip'] = df['elevation'] * df['precipitation']
    df['slope_precip'] = df['slope'] * df['precipitation']
    df['elevation_squared'] = df['elevation'] ** 2
    df['recharge_potential'] = df['soil_moisture'] * df['precipitation']

    # Temporal lagged features (keep all rows for small borehole datasets)
    if 'precipitation' in df.columns:
        precip = pd.to_numeric(df['precipitation'], errors='coerce')
        precip_median = precip.median()
        if pd.isna(precip_median):
            precip_median = 0.0
        precip = precip.fillna(precip_median)
        df['precipitation'] = precip

        if len(df) > temporal_lags:
            for lag in range(1, temporal_lags + 1):
                lag_col = f'precipitation_t-{lag}'
                df[lag_col] = precip.shift(lag)
                df[lag_col] = df[lag_col].fillna(precip_median)
        else:
            for lag in range(1, temporal_lags + 1):
                df[f'precipitation_t-{lag}'] = precip

    # Apply Gaussian filter for noise reduction
    for col in ['sbgi', 'sbgi_geology', 'groundwater_index', 'gwi_geology', 'topo_wetness', 'recharge_potential']:
        if col in df.columns:
            df[col] = gaussian_filter(df[col].values, sigma=1)

    return df

def get_potential_color(yield_value):
    for min_y, max_y, color in POTENTIAL_BREAKS:
        if min_y <= yield_value < max_y:
            return color
    return POTENTIAL_BREAKS[0][2]


def read_first_existing_csv(candidates):
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path), path
    raise FileNotFoundError(f"None of these files exist: {candidates}")

def load_external_data():
    global DATA_SOURCES
    grace = pd.DataFrame(columns=['lat', 'lon', 'grace_anomaly'])
    smap = pd.DataFrame(columns=['lat', 'lon', 'soil_moisture'])
    dem = pd.DataFrame(columns=['lat', 'lon', 'elevation', 'slope'])
    trmm = pd.DataFrame(columns=['lat', 'lon', 'precipitation'])
    DATA_SOURCES = {}

    try:
        trmm, trmm_source = read_first_existing_csv([
            "GEE_GPM_IMERG.csv",
            "TRMM.csv",
            "GPM_IMERG.csv"
        ])
        trmm.columns = trmm.columns.str.lower()
        if '.geo' in trmm.columns:
            coords = trmm['.geo'].astype(str).str.extract(r'\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]')
            trmm['lon'] = pd.to_numeric(coords[0], errors='coerce')
            trmm['lat'] = pd.to_numeric(coords[1], errors='coerce')
        if 'mean' in trmm.columns:
            trmm = trmm.rename(columns={'mean': 'precipitation'})
            trmm = trmm[['lat', 'lon', 'precipitation']].dropna()
            trmm = trmm.groupby(['lat', 'lon'], as_index=False)['precipitation'].mean()
        DATA_SOURCES['precipitation'] = trmm_source
        print(f"Loaded precipitation data from {trmm_source}")
    except Exception as e:
        print(f"Error loading TRMM data: {e}")

    try:
        grace, grace_source = read_first_existing_csv([
            "GEE_GRACE_FO.csv",
            "GRACE.csv",
            "GRACE_FO.csv",
            "grace_central_region.csv"
        ])
        grace.columns = grace.columns.str.lower()
        if 'lwe_thickness_csr' in grace.columns:
            grace = grace.rename(columns={'lwe_thickness_csr': 'grace_anomaly'})
        grace = grace.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon'
        })
        if 'lat' not in grace.columns or 'lon' not in grace.columns:
            raise KeyError('GRACE source missing lat/lon columns')
        if 'grace_anomaly' not in grace.columns:
            raise KeyError('GRACE source missing grace_anomaly column')
        grace = grace[['lat', 'lon', 'grace_anomaly']].dropna()
        grace = grace.groupby(['lat', 'lon'], as_index=False)['grace_anomaly'].mean()
        DATA_SOURCES['grace'] = grace_source
        print(f"Loaded GRACE data from {grace_source}")
    except Exception as e:
        print(f"Error loading GRACE data: {e}")

    try:
        smap, smap_source = read_first_existing_csv([
            "GEE_SMAP_SOIL_MOISTURE.csv",
            "SMAP_SOIL_MOISTURE.csv"
        ])
        smap.columns = smap.columns.str.lower()
        soil_col = next((col for col in smap.columns
                        if 'soil' in col.lower() and 'moisture' in col.lower()), None)
        if soil_col:
            smap = smap.rename(columns={soil_col: 'soil_moisture'})[['lat', 'lon', 'soil_moisture']].dropna()
            smap = smap.groupby(['lat', 'lon'], as_index=False)['soil_moisture'].mean()
            DATA_SOURCES['soil_moisture'] = smap_source
    except Exception as e:
        print(f"Error loading SMAP data: {e}")

    try:
        dem, dem_source = read_first_existing_csv([
            "GEE_DEM.csv",
            "DEM.csv"
        ])
        dem.columns = dem.columns.str.lower()
        dem = dem.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon'
        })
        if 'elevation' not in dem.columns:
            dem['elevation'] = np.nan
        if 'slope' not in dem.columns:
            dem['slope'] = np.nan
        dem = dem[['lat', 'lon', 'elevation', 'slope']].dropna()
        dem = dem.groupby(['lat', 'lon'], as_index=False).mean(numeric_only=True)
        DATA_SOURCES['dem'] = dem_source
    except Exception as e:
        print(f"Error loading DEM data: {e}")

    return grace, smap, dem, trmm

def create_static_yield_map(grid, gdf, boreholes_subset, output_path="results/groundwater_potential_map.png"):
    central_polygon = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    points = [Point(lon, lat) for lon, lat in zip(grid['lon'], grid['lat'])]
    mask = [central_polygon.contains(point) for point in points]
    filtered_grid = grid[mask]

    x = filtered_grid['lon'].values
    y = filtered_grid['lat'].values
    z = filtered_grid['predicted_yield'].values

    cmap = ListedColormap(['#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF'])
    bounds = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 12))
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=20, alpha=0.8, edgecolor='none', label='Predicted Yield')

    # Add 30% of boreholes for validation
    test_x = boreholes_subset['lon'].values
    test_y = boreholes_subset['lat'].values
    test_z = boreholes_subset['yield'].values
    ax.scatter(test_x, test_y, c=test_z, cmap=cmap, norm=norm, marker='*', s=100, 
               edgecolor='black', label='Validation Boreholes (Actual Yield)')

    # Add elevation contour for spatial pattern analysis
    elev = filtered_grid['elevation'].values
    cs = ax.tricontour(x, y, elev, levels=10, colors='gray', linestyles='--', alpha=0.5)
    ax.clabel(cs, fmt='%1.0f', inline=True, fontsize=8)

    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            coords = []
            for poly in geom.geoms:
                coords.extend(list(poly.exterior.coords))
        else:
            continue

        poly = mplPolygon(coords, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(poly)

    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    cbar = plt.colorbar(sc, label='Groundwater Yield (m³/h)', shrink=0.7)
    cbar.set_ticks([0.25, 0.75, 1.25, 1.75, 2.25])
    cbar.set_ticklabels(['Very Low (0-0.5)', 'Low (0.5-1)', 'Moderate (1-1.5)',
                        'High (1.5-2)', 'Very High (2-2.5)'])

    plt.title('Central Region Groundwater Potential with Validation Boreholes', fontsize=16, pad=20)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')

    scale_length = 0.2 * (bounds[2] - bounds[0])
    plt.plot([bounds[0] + 0.05*(bounds[2]-bounds[0]), bounds[0] + 0.05*(bounds[2]-bounds[0]) + scale_length],
             [bounds[1] + 0.05*(bounds[3]-bounds[1]), bounds[1] + 0.05*(bounds[3]-bounds[1])],
             color='black', linewidth=3)
    plt.text(bounds[0] + 0.05*(bounds[2]-bounds[0]) + scale_length/2,
             bounds[1] + 0.07*(bounds[3]-bounds[1]),
             f'{scale_length*111:.0f} km', ha='center', fontsize=10)

    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='black'))
    ax.annotate('', xy=(0.95, 0.90), xycoords='axes fraction',
                xytext=(0.95, 0.95), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved to {output_path}")

def create_geotiff(grid, gdf, output_path="results/groundwater_potential.tif"):
    central_polygon = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    points = [Point(lon, lat) for lon, lat in zip(grid['lon'], grid['lat'])]
    mask = [central_polygon.contains(point) for point in points]
    filtered_grid = grid[mask]

    x = np.linspace(min(filtered_grid['lon']), max(filtered_grid['lon']), 500)
    y = np.linspace(min(filtered_grid['lat']), max(filtered_grid['lat']), 500)
    xx, yy = np.meshgrid(x, y)

    zz = griddata((filtered_grid['lon'], filtered_grid['lat']),
                 filtered_grid['predicted_yield'],
                 (xx, yy), method='linear')

    res_x = (x[-1] - x[0]) / len(x)
    res_y = (y[-1] - y[0]) / len(y)

    transform = from_origin(x[0] - res_x/2, y[-1] + res_y/2, res_x, res_y)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=zz.shape[0],
        width=zz.shape[1],
        count=1,
        dtype=zz.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(zz, 1)

    print(f"GeoTIFF saved to {output_path}")

def find_optimal_classes(yield_values):
    yield_values = yield_values.values.reshape(-1, 1)
    best_score = -1
    best_n = 2

    for n in range(2, 6):
        kmeans = KMeans(n_clusters=n, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(yield_values)
        score = silhouette_score(yield_values, labels)
        if score > best_score:
            best_score = score
            best_n = n

    kmeans = KMeans(n_clusters=best_n, random_state=RANDOM_STATE)
    kmeans.fit(yield_values)
    centers = sorted(kmeans.cluster_centers_.flatten())
    breaks = [(0, centers[0])]
    for i in range(1, len(centers)):
        breaks.append((centers[i-1], centers[i]))
    breaks.append((centers[-1], yield_values.max()))

    return breaks

def create_interactive_map(grid, gdf, boreholes_subset, output_path="results/groundwater_potential_map.html"):
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer(
        'OpenStreetMap',
        name='Street Map',
        attr='OpenStreetMap contributors',
        max_native_zoom=19,
        max_zoom=22
    ).add_to(m)

    folium.TileLayer(
        'CartoDB positron',
        name='Light Map',
        attr='CartoDB',
        max_native_zoom=19,
        max_zoom=22
    ).add_to(m)

    gw_group = folium.FeatureGroup(name='Groundwater Potential', show=True)

    if len(grid) > 1000:
        heat_data = [[row['lat'], row['lon'], row['predicted_yield']] for _, row in grid.iterrows()]
        HeatMap(
            heat_data,
            radius=10,
            blur=20,
            min_opacity=0.3,
            max_zoom=15
        ).add_to(gw_group)
    else:
        for _, row in grid.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color=get_potential_color(row['predicted_yield']),
                fill=True,
                fill_color=get_potential_color(row['predicted_yield']),
                fill_opacity=0.8,
                weight=1,
                popup=folium.Popup(
                    f"<b>Location:</b> {row.get('place_name', 'Unknown')}<br>"
                    f"<b>Yield:</b> {row['predicted_yield']:.2f} m³/h<br>"
                    f"<b>Coordinates:</b> {row['lat']:.4f}, {row['lon']:.4f}",
                    max_width=250
                )
            ).add_to(gw_group)

    # Add validation boreholes group
    validation_group = folium.FeatureGroup(name='Validation Boreholes', show=True)
    for _, row in boreholes_subset.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=get_potential_color(row['yield']),
            fill=True,
            fill_color=get_potential_color(row['yield']),
            fill_opacity=1.0,
            weight=1,
            popup=folium.Popup(
                f"<b>Actual Yield:</b> {row['yield']:.2f} m³/h<br>"
                f"<b>Coordinates:</b> {row['lat']:.4f}, {row['lon']:.4f}",
                max_width=250
            )
        ).add_to(validation_group)

    available_fields = list(gdf.columns)
    tooltip_fields = available_fields[:2]

    folium.GeoJson(
        gdf,
        name='Region Boundary',
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5,
            'opacity': 0.7,
            'dashArray': '5, 5'
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[f'{field}:' for field in tooltip_fields],
            localize=True
        )
    ).add_to(gw_group)

    gw_group.add_to(m)
    validation_group.add_to(m)

    folium.LayerControl(
        collapsed=False,
        position='topright',
        autoZIndex=False
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid black;
                font-family: Arial; font-size: 12px; max-width: 200px;">
        <div style="font-weight: bold; margin-bottom: 5px; text-align: center;">
            Groundwater Potential Zones</div>
        <div>
            <span style="background-color: #FF0000; display: inline-block; width: 20px; height: 15px;"></span> Very Low (0-0.5)<br/>
            <hr style="margin: 5px 0;">
            <span style="background-color: #FFA500; display: inline-block; width: 20px; height: 15px;"></span> Moderate (0.5-1.5)<br/>
            <hr style="margin: 5px 0;">
            <span style="background-color: #00FF00; display: inline-block; width: 20px; height: 15px;"></span> High (1.5-2)<br/>
            <hr style="margin: 5px 0;">
            <span style="background-color: #0000FF; display: inline-block; width: 20px; height: 15px;"></span> Very High (2-2.5 and above)<br/>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    title_html = '''
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                   background-color: white; padding: 10px; border: 2px solid black;
                   font-family: Arial; font-size: 14px; max-width: 300px;">
            <b>Central Region Groundwater Potential with Validation Boreholes</b><br>
            <span style="font-size: 12px;">Click markers for details | Toggle layers in top-right</span>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    Fullscreen(
        position='topleft',
        title='Expand me',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)

    MeasureControl(
        position='bottomleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='hectares'
    ).add_to(m)

    m.save(output_path)
    print(f"Interactive map saved to {output_path}")

def train_enhanced_model(X, y, spatial_groups):
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    )

    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    unique_groups = np.unique(spatial_groups)
    n_splits = max(2, min(5, len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    print("Training base models with spatial grouping...")
    rf_scores = cross_val_score(rf, X, y, cv=gkf, groups=spatial_groups,
                                scoring='neg_mean_squared_error')
    xgb_scores = cross_val_score(xgb, X, y, cv=gkf, groups=spatial_groups,
                                 scoring='neg_mean_squared_error')

    print(f"Random Forest CV MSE: {-rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    print(f"XGBoost CV MSE: {-xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")

    rf.fit(X, y)
    xgb.fit(X, y)

    rf_pred = rf.predict(X)
    xgb_pred = xgb.predict(X)
    X_stacked = np.column_stack([rf_pred, xgb_pred])

    print("Training meta-model...")
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(final_model, X_stacked, y, cv=gkf,
                                groups=spatial_groups,
                                scoring='neg_mean_squared_error')
    final_model.fit(X_stacked, y)

    model = StackedModel(rf, xgb, final_model)
    return model, cv_scores

def calculate_prediction_intervals(model, X, residuals, n_bootstrap=100):
    base_pred = model.predict(X)
    rng = np.random.default_rng(RANDOM_STATE)
    residuals = np.asarray(residuals)
    preds = []
    for _ in range(n_bootstrap):
        sampled_residuals = rng.choice(residuals, size=len(base_pred), replace=True)
        preds.append(base_pred + sampled_residuals)

    preds = np.asarray(preds)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return lower, upper


def compute_feature_importance_table(model, features):
    rf_importances = model.model1.feature_importances_
    xgb_importances = model.model2.feature_importances_
    meta_importances = model.final_estimator_.feature_importances_
    importances = (rf_importances * meta_importances[0] + xgb_importances * meta_importances[1])
    return pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

def create_feature_contribution_chart(model, features, output_path="results/feature_contribution_chart.png"):
    plt.figure(figsize=(12, 8))
    
    feat_importance = compute_feature_importance_table(model, features)
    
    top_features = feat_importance.head(15)
    
    plt.barh(range(len(top_features)), top_features['Importance'], align='center')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance Score')
    plt.title('Top Feature Contributions to Groundwater Yield Prediction')
    plt.gca().invert_yaxis()
    
    for i, v in enumerate(top_features['Importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature contribution chart saved to {output_path}")


def write_reproducibility_manifest(
    profile_name,
    metrics,
    features,
    boreholes_file='Boreholes.csv',
    output_path="results/reproducibility_manifest.json"
):
    input_files = [
        boreholes_file,
        DATA_SOURCES.get('precipitation', 'TRMM.csv'),
        DATA_SOURCES.get('grace', 'GRACE.csv'),
        DATA_SOURCES.get('soil_moisture', 'SMAP_SOIL_MOISTURE.csv'),
        DATA_SOURCES.get('dem', 'DEM.csv'),
        'central_region_ghana.shp'
    ]
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'profile': profile_name,
        'random_state': RANDOM_STATE,
        'data_sources': DATA_SOURCES,
        'features': features,
        'metrics': metrics,
        'input_files': [collect_file_metadata(path) for path in input_files]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Reproducibility manifest saved to {output_path}")


def write_paper_summary(metrics, top_features, profile_name, output_path="results/research_report.txt"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Groundwater Yield Prediction - Central Region, Ghana\n")
        f.write("=================================================\n\n")
        f.write(f"Run profile: {profile_name}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Data sources: {DATA_SOURCES}\n\n")
        f.write("SBGI Concept Implemented\n")
        f.write("------------------------\n")
        f.write("SBGI (Satellite-Based Groundwater Index) is implemented as:\n")
        f.write("SBGI = ((GRACE anomaly + precipitation) * sqrt(soil moisture)) / (elevation + 1e-6)\n")
        f.write("SBGI-geology interaction is implemented as:\n")
        f.write("SBGI_geology = SBGI * (0.8 + 0.4 * geo_cluster)\n\n")
        f.write("These features are used directly in prediction: sbgi, sbgi_geology.\n")
        f.write("Legacy aliases are preserved for compatibility: groundwater_index, gwi_geology.\n\n")
        f.write("Model Metrics\n")
        f.write("-------------\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")
        f.write("\nTop Feature Importances\n")
        f.write("-----------------------\n")
        for _, row in top_features.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.6f}\n")
    print(f"Paper summary saved to {output_path}")

def create_yield_heatmap(grid, gdf, output_path="results/yield_heatmap.png"):
    plt.figure(figsize=(14, 12))
    
    central_polygon = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    points = [Point(lon, lat) for lon, lat in zip(grid['lon'], grid['lat'])]
    mask = [central_polygon.contains(point) for point in points]
    filtered_grid = grid[mask]
    
    hb = plt.hexbin(filtered_grid['lon'], filtered_grid['lat'], 
                   C=filtered_grid['predicted_yield'], 
                   gridsize=50, cmap='viridis', reduce_C_function=np.mean)
    
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            coords = []
            for poly in geom.geoms:
                coords.extend(list(poly.exterior.coords))
        else:
            continue
        
        poly = mplPolygon(coords, fill=False, edgecolor='black', linewidth=2)
        plt.gca().add_patch(poly)
    
    bounds = gdf.total_bounds
    plt.xlim(bounds[0], bounds[2])
    plt.ylim(bounds[1], bounds[3])
    
    cb = plt.colorbar(hb, label='Predicted Groundwater Yield (m³/h)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Predicted Groundwater Yield Distribution Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Yield heatmap saved to {output_path}")


def build_spatial_interpolator(dataframe, value_col):
    points = dataframe[['lon', 'lat']].to_numpy()
    values = dataframe[value_col].to_numpy()
    nearest_interp = NearestNDInterpolator(points, values)

    # Delaunay-based interpolation is prohibitively slow for very large support sets.
    if len(points) > 25000:
        def _nearest_only(lon_series, lat_series):
            return np.asarray(nearest_interp(np.asarray(lon_series), np.asarray(lat_series)), dtype=float)
        return _nearest_only

    linear_interp = LinearNDInterpolator(points, values)

    def _interp(lon_series, lat_series):
        lon_arr = np.asarray(lon_series)
        lat_arr = np.asarray(lat_series)
        linear_vals = linear_interp(lon_arr, lat_arr)
        if np.isscalar(linear_vals):
            linear_vals = np.array([linear_vals])
        linear_vals = np.asarray(linear_vals, dtype=float)
        nan_mask = np.isnan(linear_vals)
        if nan_mask.any():
            nearest_vals = nearest_interp(lon_arr[nan_mask], lat_arr[nan_mask])
            linear_vals[nan_mask] = nearest_vals
        return linear_vals

    return _interp

def add_interpolated_features(df):
    global interpolators
    for feature, interp in interpolators.items():
        df[feature] = interp(df['lon'], df['lat'])
    return df


def build_spatial_groups(df):
    if 'district_new' in df.columns:
        district = df['district_new'].fillna('unknown').astype(str)
        return pd.factorize(district)[0]
    return df['spatial_group'].to_numpy()


def build_laplacian_from_coords(coords, k_neighbors=5):
    n = len(coords)
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    k = max(1, min(k_neighbors, n - 1))
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)
    sigma = np.nanmedian(distances[:, 1:])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            w = np.exp(-((d ** 2) / (2 * sigma ** 2)))
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)

    D = np.diag(A.sum(axis=1))
    return D - A


def mode_or_first(series):
    mode_vals = series.mode(dropna=True)
    if not mode_vals.empty:
        return mode_vals.iloc[0]
    return series.iloc[0] if len(series) else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZED MODEL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_enhanced_sbgwi(df):
    """
    Multi-scale, physics-expanded SBGWI variants.
    All computed without target leakage — purely from satellite covariates.
    """
    g = df['grace_anomaly'].clip(lower=0)
    p = df['precipitation'].clip(lower=0)
    sm = df['soil_moisture'].clip(lower=0)
    elev = df['elevation'].clip(lower=1)
    sl = df['slope'].clip(lower=0.01)

    # Core SBGWI (existing)
    sbgwi_base = ((g + p) * np.sqrt(sm)) / elev

    # Variant 1 – log-scaled recharge ratio: dampens extreme elevation effects
    sbgwi_log = (np.log1p(g) + np.log1p(p)) * np.sqrt(sm) / np.log1p(elev)

    # Variant 2 – precipitation-weighted: emphasises infiltration availability
    sbgwi_precip = (p ** 2 * sm) / (elev * (sl + 0.1))

    # Variant 3 – GRACE dominance index: storage-limited recharge signal
    sbgwi_grace = g * sm / np.log1p(elev + sl)

    # Variant 4 – composite recharge efficiency (TWI × SBGWI interaction)
    twi = np.log(np.maximum(p, 1e-6) / (sl + 0.1))
    sbgwi_topo = sbgwi_base * np.clip(twi, 0, None)

    # Multiplicative geology interaction (more expressive than linear)
    geo = df['geo_cluster'].clip(lower=0)
    sbgwi_geo_mult = sbgwi_base * (1.0 + 0.5 * geo) ** 2

    # Normalize all to median absolute deviation scale (rank-stable)
    def _robust_scale(x):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return (x - med) / (mad + 1e-9)

    df['sbgwi_log']       = _robust_scale(sbgwi_log)
    df['sbgwi_precip']    = _robust_scale(sbgwi_precip)
    df['sbgwi_grace']     = _robust_scale(sbgwi_grace)
    df['sbgwi_topo']      = _robust_scale(sbgwi_topo)
    df['sbgwi_geo_mult']  = _robust_scale(sbgwi_geo_mult)
    # Keep original sbgi for backward compatibility
    return df


def compute_spatial_lag_features(train_coords, train_yields_log, val_coords,
                                   k=5, bandwidth=None, exclude_self=False):
    """
    Inverse-distance-weighted spatial lag features computed inside a CV fold.
    These are NOT leakage: training and validation points are disjoint.
    Returns:
        idw_yield_log  – spatial interpolation of log-yield at validation coords
        log_nearest_dist – log of nearest training-productive distance
    """
    n = len(train_coords)
    k_eff = min(k + 1, n) if exclude_self and n > 1 else min(k, n)
    if k_eff == 0 or len(val_coords) == 0:
        nv = len(val_coords)
        return np.zeros(nv), np.full(nv, np.log1p(1.0))

    tree = cKDTree(train_coords)
    distances, indices = tree.query(val_coords, k=k_eff)

    if k_eff == 1:
        distances = distances[:, np.newaxis]
        indices   = indices[:, np.newaxis]

    if exclude_self and np.array_equal(np.asarray(val_coords), np.asarray(train_coords)) and k_eff > 1:
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        self_match = np.isclose(distances[:, 0], 0.0)
        if np.any(self_match):
            distances = np.where(self_match[:, np.newaxis], distances[:, 1:], distances[:, :k])
            indices = np.where(self_match[:, np.newaxis], indices[:, 1:], indices[:, :k])
            if distances.shape[1] == 0:
                distances = np.full((len(val_coords), 1), np.inf)
                indices = np.zeros((len(val_coords), 1), dtype=int)

    # adaptive bandwidth: median nearest-neighbor distance in training set
    if bandwidth is None:
        _, self_d = tree.query(train_coords, k=min(3, n))
        if self_d.ndim == 1:
            self_d = self_d[:, np.newaxis]
        bandwidth = float(np.nanmedian(self_d[:, -1])) + 1e-6

    w = 1.0 / np.maximum(distances, 1e-8) ** 2
    w /= w.sum(axis=1, keepdims=True)
    idw_yield_log = (w * train_yields_log[indices]).sum(axis=1)

    # Damp spatial-lag influence for far extrapolation to improve transfer across districts.
    global_prior = float(np.nanmedian(train_yields_log))
    distance_ratio = distances[:, 0] / (bandwidth + 1e-9)
    shrink = np.exp(-(distance_ratio ** 2))
    idw_yield_log = shrink * idw_yield_log + (1.0 - shrink) * global_prior

    log_nearest_dist = np.log1p(distances[:, 0])
    return idw_yield_log, log_nearest_dist


def compute_district_log_stats(train_districts, train_log_y):
    """Build train-only district log-yield statistics for leakage-safe adaptation."""
    train_districts = np.asarray(train_districts, dtype=str)
    train_log_y = np.asarray(train_log_y, dtype=float)

    global_median = float(np.nanmedian(train_log_y)) if len(train_log_y) else 0.0
    global_iqr = float(np.nanpercentile(train_log_y, 75) - np.nanpercentile(train_log_y, 25)) if len(train_log_y) else 0.0

    stats = {
        'global_median': global_median,
        'global_iqr': global_iqr,
        'district': {}
    }
    if len(train_districts) == 0:
        return stats

    for d in np.unique(train_districts):
        vals = train_log_y[train_districts == d]
        if len(vals) == 0:
            continue
        d_med = float(np.nanmedian(vals))
        d_iqr = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        stats['district'][str(d)] = {'median': d_med, 'iqr': d_iqr}
    return stats


def district_stats_to_features(target_districts, stats):
    """Map districts to train-only median/IQR features with global fallback."""
    if target_districts is None:
        n = 0
    else:
        n = len(target_districts)

    if n == 0:
        return np.zeros(0), np.zeros(0)

    dvals = np.asarray(target_districts, dtype=str)
    g_med = float(stats.get('global_median', 0.0))
    g_iqr = float(stats.get('global_iqr', 0.0))
    lookup = stats.get('district', {})

    med = np.full(n, g_med, dtype=float)
    iqr = np.full(n, g_iqr, dtype=float)
    for i, d in enumerate(dvals):
        if d in lookup:
            med[i] = float(lookup[d].get('median', g_med))
            iqr[i] = float(lookup[d].get('iqr', g_iqr))
    return med, iqr


def fit_stage2_base_learners(X_train, y_train):
    """RF + XGBoost + LightGBM (if available) base learners."""
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb = XGBRegressor(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.3,
        min_child_weight=5,
        objective='reg:pseudohubererror',
        eval_metric='mae',
        random_state=RANDOM_STATE,
        verbosity=0
    )
    rf.fit(X_train, y_train)
    try:
        xgb.fit(X_train, y_train)
    except Exception:
        xgb = XGBRegressor(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            reg_alpha=0.3,
            min_child_weight=5,
            objective='reg:squarederror',
            eval_metric='mae',
            random_state=RANDOM_STATE,
            verbosity=0
        )
        xgb.fit(X_train, y_train)

    models = [rf, xgb]
    if HAS_LIGHTGBM:
        lgb = LGBMRegressor(
            n_estimators=700,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            reg_alpha=0.3,
            min_child_samples=5,
            objective='huber',
            alpha=0.9,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
        try:
            lgb.fit(X_train, y_train)
        except Exception:
            lgb = LGBMRegressor(
                n_estimators=700,
                num_leaves=15,
                learning_rate=0.05,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                reg_alpha=0.3,
                min_child_samples=5,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            )
            lgb.fit(X_train, y_train)
        models.append(lgb)

    return models


def tune_spatial_ridge(X_meta, y, coords, alphas=(0.1, 1.0, 5.0), gammas=(0.1, 1.0, 5.0)):
    """Grid search best alpha/gamma for Spatial-Ridge using LOOCV on small n."""
    best_mse = np.inf
    best_alpha, best_gamma = 1.0, 1.0
    n = len(y)
    if n <= 4:
        return best_alpha, best_gamma

    for alpha in alphas:
        for gamma in gammas:
            errs = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                X_tr = X_meta[mask]
                y_tr = y[mask]
                c_tr = coords[mask]
                X_va = X_meta[[i]]
                y_va = y[i]
                try:
                    meta = SpatialRidgeMetaLearner(alpha=alpha, gamma=gamma, k_neighbors=min(5, mask.sum()-1))
                    meta.fit(X_tr, y_tr, c_tr)
                    pred = meta.predict(X_va)[0]
                    errs.append((pred - y_va) ** 2)
                except Exception:
                    errs.append(np.inf)
            mse = np.mean(errs)
            if mse < best_mse:
                best_mse = mse
                best_alpha, best_gamma = alpha, gamma
    return best_alpha, best_gamma


class GPWrapper:
    """Top-level picklable wrapper around a fitted GaussianProcessRegressor."""
    def __init__(self, gp, coords_mean, coords_std):
        self._gp = gp
        self._cmean = coords_mean
        self._cstd = coords_std

    def predict(self, X_meta_new, coords_new=None):
        if coords_new is None:
            pad = np.zeros((len(X_meta_new), 2))
        else:
            pad = (np.asarray(coords_new) - self._cmean) / (self._cstd + 1e-9)
        Xp = np.hstack([np.asarray(X_meta_new), pad])
        return self._gp.predict(Xp)


def predict_meta(meta, meta_type, X_meta_new, coords_new):
    if meta_type == 'GP':
        return meta.predict(X_meta_new, coords_new)
    return meta.predict(X_meta_new)


def fit_gp_meta_learner(X_meta, y, coords):
    """
    Gaussian Process meta-learner combining base learner predictions and
    spatial coordinates. Uses Matérn 3/2 kernel for smooth spatial trends.
    Falls back to Spatial Ridge if GP is unavailable.
    """
    if not HAS_GP:
        meta = SpatialRidgeMetaLearner(alpha=1.0, gamma=1.0, k_neighbors=5)
        meta.fit(X_meta, y, coords)
        return meta, 'SpatialRidge'

    n = len(y)
    coords_norm = coords - coords.mean(axis=0)
    coords_norm /= (coords_norm.std(axis=0) + 1e-9)
    X_gp = np.hstack([X_meta, coords_norm])

    length_scale_init = np.ones(X_gp.shape[1])
    kernel = Matern(length_scale=length_scale_init, nu=1.5) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=RANDOM_STATE
    )
    try:
        gp.fit(X_gp, y)

        return GPWrapper(gp, coords.mean(axis=0), coords.std(axis=0) + 1e-9), 'GP'
    except Exception as e:
        print(f"  GP fit failed ({e}), falling back to Spatial Ridge")
        meta = SpatialRidgeMetaLearner(alpha=1.0, gamma=1.0, k_neighbors=5)
        meta.fit(X_meta, y, coords)
        return meta, 'SpatialRidge'


def fit_spatial_ridge_meta_learner(X_meta, y, coords, alpha=1.0, gamma=1.0):
    k_neighbors = max(2, min(5, len(y) - 1)) if len(y) > 2 else 1
    meta = SpatialRidgeMetaLearner(alpha=alpha, gamma=gamma, k_neighbors=k_neighbors)
    meta.fit(X_meta, y, coords)
    return meta, 'SpatialRidge'


def compute_meta_selection_score(metrics):
    """Composite selection score aligned with publication utility metrics."""
    rho = np.nan_to_num(metrics.get('spearman_r', np.nan), nan=0.0)
    zone = np.nan_to_num(metrics.get('zone_accuracy', np.nan), nan=0.0)
    direction = np.nan_to_num(metrics.get('direction_accuracy', np.nan), nan=0.0)
    rmse = metrics.get('rmse', np.inf)
    rmse_penalty = np.nan_to_num(rmse, nan=2.0, posinf=2.0)
    return float(0.60 * rho + 0.25 * zone + 0.15 * direction - 0.08 * rmse_penalty)


def compute_meta_selection_utility(metrics):
    """Utility-only score used with RMSE guardrails when comparing meta candidates."""
    rho = np.nan_to_num(metrics.get('spearman_r', np.nan), nan=0.0)
    zone = np.nan_to_num(metrics.get('zone_accuracy', np.nan), nan=0.0)
    direction = np.nan_to_num(metrics.get('direction_accuracy', np.nan), nan=0.0)
    return float(0.60 * rho + 0.25 * zone + 0.15 * direction)


def select_meta_model_spatial_cv(X_meta, y, coords, groups, ridge_alpha=1.0, ridge_gamma=1.0,
                                 eval_transform=None):
    """
    Choose between GP and Spatial Ridge using inner GroupKFold on meta-features.
    Returns fitted meta model on full data, chosen type, and CV summary metrics.
    """
    X_meta = np.asarray(X_meta)
    y = np.asarray(y)
    coords = np.asarray(coords)
    groups = np.asarray(groups)

    candidates = ['SpatialRidge'] + (['GP'] if HAS_GP else [])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        if HAS_GP:
            meta, meta_type = fit_gp_meta_learner(X_meta, y, coords)
            if meta_type == 'GP':
                return meta, 'GP', {'cv_rmse_gp': np.nan, 'cv_rmse_ridge': np.nan}
        meta, _ = fit_spatial_ridge_meta_learner(X_meta, y, coords, ridge_alpha, ridge_gamma)
        return meta, 'SpatialRidge', {'cv_rmse_gp': np.nan, 'cv_rmse_ridge': np.nan}

    n_splits = max(2, min(4, len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    scores = {c: {'rmse': [], 'spearman': [], 'zone': [], 'direction': [], 'score': [], 'utility': []} for c in candidates}

    for tr_idx, va_idx in gkf.split(X_meta, y, groups):
        X_tr, X_va = X_meta[tr_idx], X_meta[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        c_tr, c_va = coords[tr_idx], coords[va_idx]

        # Spatial Ridge candidate
        try:
            meta_r, _ = fit_spatial_ridge_meta_learner(X_tr, y_tr, c_tr, ridge_alpha, ridge_gamma)
            pred_r = predict_meta(meta_r, 'SpatialRidge', X_va, c_va)
            y_va_eval = eval_transform(y_va) if eval_transform is not None else y_va
            pred_r_eval = eval_transform(pred_r) if eval_transform is not None else pred_r
            m_r = compute_publishable_metrics(y_va_eval, pred_r_eval)
            scores['SpatialRidge']['rmse'].append(m_r.get('rmse', np.inf))
            scores['SpatialRidge']['spearman'].append(m_r.get('spearman_r', np.nan))
            scores['SpatialRidge']['zone'].append(m_r.get('zone_accuracy', np.nan))
            scores['SpatialRidge']['direction'].append(m_r.get('direction_accuracy', np.nan))
            scores['SpatialRidge']['score'].append(compute_meta_selection_score(m_r))
            scores['SpatialRidge']['utility'].append(compute_meta_selection_utility(m_r))
        except Exception:
            scores['SpatialRidge']['rmse'].append(np.inf)
            scores['SpatialRidge']['spearman'].append(np.nan)
            scores['SpatialRidge']['zone'].append(np.nan)
            scores['SpatialRidge']['direction'].append(np.nan)
            scores['SpatialRidge']['score'].append(-np.inf)
            scores['SpatialRidge']['utility'].append(-np.inf)

        if HAS_GP:
            try:
                meta_g, meta_t = fit_gp_meta_learner(X_tr, y_tr, c_tr)
                if meta_t != 'GP':
                    raise RuntimeError('GP unavailable in this fold')
                pred_g = predict_meta(meta_g, 'GP', X_va, c_va)
                y_va_eval = eval_transform(y_va) if eval_transform is not None else y_va
                pred_g_eval = eval_transform(pred_g) if eval_transform is not None else pred_g
                m_g = compute_publishable_metrics(y_va_eval, pred_g_eval)
                scores['GP']['rmse'].append(m_g.get('rmse', np.inf))
                scores['GP']['spearman'].append(m_g.get('spearman_r', np.nan))
                scores['GP']['zone'].append(m_g.get('zone_accuracy', np.nan))
                scores['GP']['direction'].append(m_g.get('direction_accuracy', np.nan))
                scores['GP']['score'].append(compute_meta_selection_score(m_g))
                scores['GP']['utility'].append(compute_meta_selection_utility(m_g))
            except Exception:
                scores['GP']['rmse'].append(np.inf)
                scores['GP']['spearman'].append(np.nan)
                scores['GP']['zone'].append(np.nan)
                scores['GP']['direction'].append(np.nan)
                scores['GP']['score'].append(-np.inf)
                scores['GP']['utility'].append(-np.inf)

    rmse_r = float(np.nanmean(scores['SpatialRidge']['rmse'])) if scores['SpatialRidge']['rmse'] else np.inf
    rho_r = float(np.nanmean(scores['SpatialRidge']['spearman'])) if scores['SpatialRidge']['spearman'] else np.nan
    zone_r = float(np.nanmean(scores['SpatialRidge']['zone'])) if scores['SpatialRidge']['zone'] else np.nan
    dir_r = float(np.nanmean(scores['SpatialRidge']['direction'])) if scores['SpatialRidge']['direction'] else np.nan
    score_r = float(np.nanmean(scores['SpatialRidge']['score'])) if scores['SpatialRidge']['score'] else -np.inf
    utility_r = float(np.nanmean(scores['SpatialRidge']['utility'])) if scores['SpatialRidge']['utility'] else -np.inf

    rmse_g = np.inf
    rho_g = np.nan
    zone_g = np.nan
    dir_g = np.nan
    score_g = -np.inf
    utility_g = -np.inf
    if HAS_GP and 'GP' in scores:
        rmse_g = float(np.nanmean(scores['GP']['rmse'])) if scores['GP']['rmse'] else np.inf
        rho_g = float(np.nanmean(scores['GP']['spearman'])) if scores['GP']['spearman'] else np.nan
        zone_g = float(np.nanmean(scores['GP']['zone'])) if scores['GP']['zone'] else np.nan
        dir_g = float(np.nanmean(scores['GP']['direction'])) if scores['GP']['direction'] else np.nan
        score_g = float(np.nanmean(scores['GP']['score'])) if scores['GP']['score'] else -np.inf
        utility_g = float(np.nanmean(scores['GP']['utility'])) if scores['GP']['utility'] else -np.inf

    rmse_guard = np.inf
    rmse_guard_ok = False
    if np.isfinite(rmse_g) and np.isfinite(rmse_r):
        rmse_guard = max(0.03, 0.03 * rmse_r)
        rmse_guard_ok = (rmse_g - rmse_r) <= rmse_guard

    choose_gp = HAS_GP and (
        rmse_guard_ok and (
            (utility_g > utility_r + 0.015) or
            (np.isfinite(score_g) and np.isfinite(score_r) and score_g > score_r + 1e-6)
        )
    )

    if choose_gp:
        meta, meta_type = fit_gp_meta_learner(X_meta, y, coords)
        if meta_type != 'GP':
            meta, meta_type = fit_spatial_ridge_meta_learner(X_meta, y, coords, ridge_alpha, ridge_gamma)
    else:
        meta, meta_type = fit_spatial_ridge_meta_learner(X_meta, y, coords, ridge_alpha, ridge_gamma)

    return meta, meta_type, {
        'cv_rmse_gp': rmse_g,
        'cv_rmse_ridge': rmse_r,
        'cv_spearman_gp': rho_g,
        'cv_spearman_ridge': rho_r,
        'cv_zone_gp': zone_g,
        'cv_zone_ridge': zone_r,
        'cv_direction_gp': dir_g,
        'cv_direction_ridge': dir_r,
        'cv_score_gp': score_g,
        'cv_score_ridge': score_r,
        'cv_utility_gp': utility_g,
        'cv_utility_ridge': utility_r,
        'cv_rmse_guard_abs': rmse_guard,
        'cv_rmse_guard_ok': float(rmse_guard_ok)
    }


def compute_publishable_metrics(y_true, y_pred):
    """
    Extended metric suite for publication:
    R², RMSE, MAE, Spearman ρ, bias, and 3-class zone accuracy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {}

    r2    = float(r2_score(y_true, y_pred))  if n >= 2 else np.nan
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    bias  = float(np.mean(y_pred - y_true))

    sp_r, sp_p = spearmanr(y_true, y_pred) if n >= 4 else (np.nan, np.nan)

    # 3-zone classification accuracy: Low(<0.9), Medium(0.9-2.0), High(≥2.0)
    def _zone(v):
        return np.where(v < 0.9, 0, np.where(v < 2.0, 1, 2))

    zone_acc = float(accuracy_score(_zone(y_true), _zone(y_pred))) if n >= 2 else np.nan

    # Binary direction accuracy: does model rank wells correctly above/below median?
    med = np.median(y_true)
    dir_acc = float(np.mean((y_true >= med) == (y_pred >= med))) if n >= 2 else np.nan

    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias,
        'spearman_r': float(sp_r) if np.isfinite(sp_r) else np.nan,
        'spearman_p': float(sp_p) if np.isfinite(sp_p) else np.nan,
        'zone_accuracy': zone_acc,
        'direction_accuracy': dir_acc,
        'n': int(n)
    }


def run_optimized_training(boreholes, features):
    """
    Optimized two-stage framework for publication:
    Stage 1  – RF classifier with isotonic probability calibration.
    Stage 2  – RF + XGBoost + LightGBM base learners;
               GP (Matérn) or tuned Spatial-Ridge meta-learner;
               + spatial lag covariate inside each CV fold.
    Enhanced SBGWI variants included as additional features.
    """
    train_df, dedupe_stats = deduplicate_boreholes_for_modeling(boreholes)
    train_df = compute_enhanced_sbgwi(train_df)

    # Extended feature list: original + enhanced SBGWI variants
    extra_sbgwi = ['sbgwi_log', 'sbgwi_precip', 'sbgwi_grace', 'sbgwi_topo', 'sbgwi_geo_mult']
    all_features = features + [f for f in extra_sbgwi if f not in features]
    # Make sure all extra cols exist (they will, from compute_enhanced_sbgwi above)
    X_all_raw = train_df[all_features].copy()
    y_all = train_df['yield'].to_numpy(dtype=float)
    y_class = (y_all > 0).astype(int)
    coords_all = train_df[['lon', 'lat']].to_numpy(dtype=float)
    groups_all = build_spatial_groups(train_df)
    districts_all = (
        train_df['district_new'].fillna('unknown').astype(str).to_numpy()
        if 'district_new' in train_df.columns
        else np.array(['unknown'] * len(train_df))
    )

    unique_groups = np.unique(groups_all)
    n_splits = max(2, min(5, len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)

    stage1_oof = np.zeros(len(train_df), dtype=int)
    stage1_prob = np.zeros(len(train_df), dtype=float)
    stage2_oof = np.full(len(train_df), np.nan, dtype=float)
    stage2_oof_base = np.full(len(train_df), np.nan, dtype=float)
    resid_stage2 = np.full(len(train_df), np.nan, dtype=float)
    resid_stage1_prob = np.full(len(train_df), np.nan, dtype=float)
    resid_dmed = np.full(len(train_df), np.nan, dtype=float)
    resid_diqr = np.full(len(train_df), np.nan, dtype=float)
    resid_lag = np.full(len(train_df), np.nan, dtype=float)
    resid_dist = np.full(len(train_df), np.nan, dtype=float)

    fold_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(
            gkf.split(X_all_raw, y_class, groups_all), start=1):

        X_tr_raw = X_all_raw.iloc[train_idx]
        X_va_raw = X_all_raw.iloc[val_idx]
        y_tr_cls = y_class[train_idx]
        y_va_cls = y_class[val_idx]

        imp1 = KNNImputer(n_neighbors=3)
        sc1  = StandardScaler()
        X_tr_1 = sc1.fit_transform(imp1.fit_transform(X_tr_raw))
        X_va_1 = sc1.transform(imp1.transform(X_va_raw))

        cls = RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        cls.fit(X_tr_1, y_tr_cls)
        stage1_oof[val_idx]  = cls.predict(X_va_1)
        stage1_prob[val_idx] = cls.predict_proba(X_va_1)[:, 1]

        prod_tr_mask = y_tr_cls == 1
        prod_va_mask = y_va_cls == 1
        if prod_tr_mask.sum() < 8 or prod_va_mask.sum() == 0:
            fold_rows.append({'fold': fold_id, 'note': 'skipped', 'n_val': 0})
            continue

        tr_prod_idx  = np.array(train_idx)[prod_tr_mask]
        va_prod_idx  = np.array(val_idx)[prod_va_mask]

        X_tr_prod_raw = X_all_raw.iloc[tr_prod_idx]
        X_va_prod_raw = X_all_raw.iloc[va_prod_idx]
        y_tr_prod     = y_all[tr_prod_idx]
        target_params = fit_stage2_target_transform(y_tr_prod, upper_quantile=0.95)
        y_tr_prod_log = transform_stage2_target(y_tr_prod, target_params)

        imp2 = KNNImputer(n_neighbors=3)
        sc2  = StandardScaler()
        X_tr_prod = sc2.fit_transform(imp2.fit_transform(X_tr_prod_raw))
        X_va_prod = sc2.transform(imp2.transform(X_va_prod_raw))

        # Spatial lag features (computed on training productive coords/yields)
        tr_coords_prod = coords_all[tr_prod_idx]
        va_coords_prod = coords_all[va_prod_idx]
        tr_districts_prod = districts_all[tr_prod_idx]
        va_districts_prod = districts_all[va_prod_idx]

        district_stats = compute_district_log_stats(tr_districts_prod, y_tr_prod_log)
        dmed_tr, diqr_tr = district_stats_to_features(tr_districts_prod, district_stats)
        dmed_va, diqr_va = district_stats_to_features(va_districts_prod, district_stats)

        lag_tr, dist_tr = compute_spatial_lag_features(
            tr_coords_prod, y_tr_prod_log, tr_coords_prod, k=5, exclude_self=True)
        lag_va, dist_va = compute_spatial_lag_features(
            tr_coords_prod, y_tr_prod_log, va_coords_prod, k=5)

        X_tr_aug = np.hstack([X_tr_prod,
                              dmed_tr[:, np.newaxis],
                              diqr_tr[:, np.newaxis],
                              lag_tr[:, np.newaxis],
                              dist_tr[:, np.newaxis]])
        X_va_aug = np.hstack([X_va_prod,
                              dmed_va[:, np.newaxis],
                              diqr_va[:, np.newaxis],
                              lag_va[:, np.newaxis],
                              dist_va[:, np.newaxis]])

        # Inner nested CV for base learner OOF on training set
        g_tr = groups_all[tr_prod_idx]
        inner_splits = max(2, min(4, len(np.unique(g_tr))))
        inner_gkf = GroupKFold(n_splits=inner_splits)

        base_oof_tr = np.zeros((len(tr_prod_idx), 2 + int(HAS_LIGHTGBM)), dtype=float)
        for i_tr, i_va in inner_gkf.split(X_tr_aug, y_tr_prod_log, g_tr):
            base_models = fit_stage2_base_learners(X_tr_aug[i_tr], y_tr_prod_log[i_tr])
            for mi, m in enumerate(base_models):
                base_oof_tr[i_va, mi] = m.predict(X_tr_aug[i_va])

        # Tune Spatial Ridge on inner OOF
        best_alpha, best_gamma = tune_spatial_ridge(
            base_oof_tr, y_tr_prod_log, tr_coords_prod)

        # Select GP vs Spatial Ridge by inner spatial CV on meta-features.
        meta, meta_type, meta_cv = select_meta_model_spatial_cv(
            base_oof_tr,
            y_tr_prod_log,
            tr_coords_prod,
            g_tr,
            ridge_alpha=best_alpha,
            ridge_gamma=best_gamma,
            eval_transform=lambda arr, tp=target_params: inverse_stage2_target(arr, tp)
        )

        # Fit final base learners on full training productive set
        base_models_full = fit_stage2_base_learners(X_tr_aug, y_tr_prod_log)
        base_val_preds = np.column_stack(
            [m.predict(X_va_aug) for m in base_models_full])

        # Meta predict
        val_pred_log = predict_meta(meta, meta_type, base_val_preds, va_coords_prod)

        stage2_oof[va_prod_idx] = inverse_stage2_target(val_pred_log, target_params)

        y_va_true = y_all[va_prod_idx]
        y_va_pred = stage2_oof[va_prod_idx]
        fold_metrics = compute_publishable_metrics(y_va_true, y_va_pred)
        fold_rows.append({
            'fold': fold_id,
            'n_val': int(len(va_prod_idx)),
            'meta_type': meta_type,
            'spatial_ridge_alpha': best_alpha,
            'spatial_ridge_gamma': best_gamma,
            'meta_cv_rmse_gp': meta_cv.get('cv_rmse_gp', np.nan),
            'meta_cv_rmse_ridge': meta_cv.get('cv_rmse_ridge', np.nan),
            'note': 'ok',
            **fold_metrics
        })
        print(f"  Fold {fold_id}: n_val={len(va_prod_idx)}, "
              f"R²={fold_metrics['r2']:.3f}, RMSE={fold_metrics['rmse']:.3f}, "
              f"ρ={fold_metrics['spearman_r']:.3f}, meta={meta_type}")

        # Store residual-corrector training features for outer OOF rows.
        stage2_oof_base[va_prod_idx] = inverse_stage2_target(val_pred_log, target_params)
        resid_stage1_prob[va_prod_idx] = stage1_prob[va_prod_idx]
        resid_dmed[va_prod_idx] = dmed_va
        resid_diqr[va_prod_idx] = diqr_va
        resid_lag[va_prod_idx] = lag_va
        resid_dist[va_prod_idx] = dist_va

    # Overall OOF metrics
    valid_mask = np.isfinite(stage2_oof)
    y_s2_true = y_all[valid_mask]
    y_s2_pred_base = stage2_oof[valid_mask]
    d_s2      = districts_all[valid_mask]
    stage1_metrics   = {
        'accuracy':  float(accuracy_score(y_class, stage1_oof)),
        'precision': float(precision_score(y_class, stage1_oof, zero_division=0)),
        'recall':    float(recall_score(y_class, stage1_oof, zero_division=0)),
        'f1':        float(f1_score(y_class, stage1_oof, zero_division=0))
    }
    stage2_calibrator = fit_stage2_linear_calibrator(y_s2_true, y_s2_pred_base)
    stage2_base_calibrated = apply_stage2_linear_calibrator(y_s2_pred_base, stage2_calibrator)
    residual_feature_matrix = np.column_stack([
        stage2_base_calibrated,
        stage1_prob[valid_mask],
        resid_dmed[valid_mask],
        resid_diqr[valid_mask],
        resid_lag[valid_mask],
        resid_dist[valid_mask]
    ])
    residual_target = y_s2_true - (stage1_prob[valid_mask] * stage2_base_calibrated)
    stage2_residual_corrector = fit_stage2_residual_corrector(residual_feature_matrix, residual_target)
    corrected_stage2_pred = apply_stage2_residual_corrector(
        stage1_prob[valid_mask] * stage2_base_calibrated,
        stage2_residual_corrector,
        residual_feature_matrix
    )
    corrected_stage2_pred = np.clip(corrected_stage2_pred, 0, None)
    y_s2_pred = crossfit_stage2_isotonic_predictions(
        y_s2_true,
        corrected_stage2_pred,
        groups_all[valid_mask]
    )
    stage2_isotonic_calibrator = fit_stage2_isotonic_calibrator(y_s2_true, corrected_stage2_pred)
    y_s2_pred = np.clip(y_s2_pred, 0, None)
    stage2_metrics = compute_publishable_metrics(y_s2_true, y_s2_pred)
    stage2_metrics_base = compute_publishable_metrics(y_s2_true, y_s2_pred_base)
    district_metrics = compute_stage2_district_metrics(y_s2_true, y_s2_pred, d_s2)
    corrected_stage2_metrics = compute_publishable_metrics(y_s2_true, corrected_stage2_pred)
    fold_metrics_df  = pd.DataFrame(fold_rows)

    create_stage1_confusion_matrix(y_class, stage1_oof)
    create_stage2_obs_pred_plot(y_s2_true, y_s2_pred)

    # --- Fit final models on full dataset ---
    all_features_for_fit = all_features  # store ref
    train_df_enriched = compute_enhanced_sbgwi(train_df)
    X_all_full = train_df_enriched[all_features].copy()
    y_all_full = train_df['yield'].to_numpy(dtype=float)
    y_cls_full = (y_all_full > 0).astype(int)

    imp1_f = KNNImputer(n_neighbors=3);  sc1_f = StandardScaler()
    X1_f   = sc1_f.fit_transform(imp1_f.fit_transform(X_all_full))
    cls_f  = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    cls_f.fit(X1_f, y_cls_full)

    prod_mask_f  = y_cls_full == 1
    X_prod_raw_f = X_all_full.iloc[prod_mask_f]
    y_prod_f     = y_all_full[prod_mask_f]
    tparams_f    = fit_stage2_target_transform(y_prod_f, upper_quantile=0.95)
    y_prod_log_f = transform_stage2_target(y_prod_f, tparams_f)
    coords_prod_f = coords_all[prod_mask_f]
    districts_prod_f = districts_all[prod_mask_f]

    imp2_f = KNNImputer(n_neighbors=3);  sc2_f = StandardScaler()
    X_prod_f = sc2_f.fit_transform(imp2_f.fit_transform(X_prod_raw_f))
    district_stats_f = compute_district_log_stats(districts_prod_f, y_prod_log_f)
    dmed_f, diqr_f = district_stats_to_features(districts_prod_f, district_stats_f)
    lag_f, dist_f = compute_spatial_lag_features(
        coords_prod_f, y_prod_log_f, coords_prod_f, k=5, exclude_self=True)
    X_prod_aug_f = np.hstack([X_prod_f,
                               dmed_f[:, np.newaxis],
                               diqr_f[:, np.newaxis],
                               lag_f[:, np.newaxis],
                               dist_f[:, np.newaxis]])

    # OOF on full for GP training
    g_prod_f = groups_all[prod_mask_f]
    outer_splits = max(2, min(5, len(np.unique(g_prod_f))))
    outer_gkf = GroupKFold(n_splits=outer_splits)
    base_oof_f = np.zeros((len(X_prod_aug_f), 2 + int(HAS_LIGHTGBM)), dtype=float)
    for i_tr, i_va in outer_gkf.split(X_prod_aug_f, y_prod_log_f, g_prod_f):
        bm = fit_stage2_base_learners(X_prod_aug_f[i_tr], y_prod_log_f[i_tr])
        for mi, m in enumerate(bm):
            base_oof_f[i_va, mi] = m.predict(X_prod_aug_f[i_va])

    best_alpha_f, best_gamma_f = tune_spatial_ridge(base_oof_f, y_prod_log_f, coords_prod_f)
    meta_f, meta_type_f, meta_cv_full = select_meta_model_spatial_cv(
        base_oof_f,
        y_prod_log_f,
        coords_prod_f,
        g_prod_f,
        ridge_alpha=best_alpha_f,
        ridge_gamma=best_gamma_f,
        eval_transform=lambda arr, tp=tparams_f: inverse_stage2_target(arr, tp)
    )
    base_models_f = fit_stage2_base_learners(X_prod_aug_f, y_prod_log_f)

    return {
        'stage1': {
            'imputer': imp1_f, 'scaler': sc1_f, 'model': cls_f,
            'metrics': stage1_metrics
        },
        'stage2': {
            'imputer': imp2_f, 'scaler': sc2_f,
            'base_models': base_models_f,
            'meta': meta_f, 'meta_type': meta_type_f,
            'meta_cv_summary': meta_cv_full,
            'spatial_ridge_alpha': best_alpha_f,
            'spatial_ridge_gamma': best_gamma_f,
            'district_stats': district_stats_f,
            'calibrator': stage2_calibrator,
            'residual_corrector': stage2_residual_corrector,
            'isotonic_calibrator': stage2_isotonic_calibrator,
            'target_params': tparams_f,
            'metrics': stage2_metrics,
            'metrics_base': stage2_metrics_base,
        },
        'spatial': {
            'train_coords_prod': coords_prod_f,
            'train_log_yields': y_prod_log_f
        },
        'all_features': all_features,
        'diagnostics': {
            'deduplication': dedupe_stats,
            'fold_metrics': fold_metrics_df,
            'district_metrics': district_metrics
        },
        'oof': {'stage2_true': y_s2_true, 'stage2_pred': y_s2_pred, 'stage2_pred_base': y_s2_pred_base}
    }


def optimized_predict_yield(bundle, X_raw_df, pred_coords):
    """Predict with optimized bundle; includes spatial lag at prediction time."""
    all_features = bundle['all_features']
    X_raw = X_raw_df.reindex(columns=all_features, fill_value=0.0).copy()

    X1 = bundle['stage1']['scaler'].transform(
        bundle['stage1']['imputer'].transform(X_raw))
    productive_prob = bundle['stage1']['model'].predict_proba(X1)[:, 1]
    productive_cls  = bundle['stage1']['model'].predict(X1)

    X2 = bundle['stage2']['scaler'].transform(
        bundle['stage2']['imputer'].transform(X_raw))

    pred_districts = None
    if 'district_new' in X_raw_df.columns:
        pred_districts = X_raw_df['district_new'].fillna('unknown').astype(str).to_numpy()
    elif 'district' in X_raw_df.columns:
        pred_districts = X_raw_df['district'].fillna('unknown').astype(str).to_numpy()
    else:
        pred_districts = np.array(['unknown'] * len(X_raw_df))

    dmed, diqr = district_stats_to_features(
        pred_districts,
        bundle['stage2'].get('district_stats', {
            'global_median': 0.0,
            'global_iqr': 0.0,
            'district': {}
        })
    )

    train_coords = bundle['spatial']['train_coords_prod']
    train_log_y  = bundle['spatial']['train_log_yields']
    lag, dist = compute_spatial_lag_features(
        train_coords, train_log_y, pred_coords, k=5)
    X2_aug = np.hstack([
        X2,
        dmed[:, np.newaxis],
        diqr[:, np.newaxis],
        lag[:, np.newaxis],
        dist[:, np.newaxis]
    ])

    base_preds = np.column_stack(
        [m.predict(X2_aug) for m in bundle['stage2']['base_models']])

    meta = bundle['stage2']['meta']
    yield_log = predict_meta(meta, bundle['stage2']['meta_type'], base_preds, pred_coords)

    yield_prod = inverse_stage2_target(yield_log, bundle['stage2']['target_params'])
    yield_prod = apply_stage2_linear_calibrator(
        yield_prod,
        bundle['stage2'].get('calibrator', {'slope': 1.0, 'intercept': 0.0})
    )
    yield_prod = np.clip(yield_prod, 0, None)

    expected_yield = productive_prob * yield_prod
    resid_model = bundle['stage2'].get('residual_corrector', None)
    if resid_model is not None:
        residual_features = np.column_stack([
            yield_prod,
            productive_prob,
            dmed,
            diqr,
            lag,
            dist
        ])
        expected_yield = apply_stage2_residual_corrector(expected_yield, resid_model, residual_features)
        expected_yield = np.clip(expected_yield, 0, None)

    expected_yield = apply_stage2_isotonic_calibrator(
        expected_yield,
        bundle['stage2'].get('isotonic_calibrator', None)
    )
    expected_yield = np.clip(expected_yield, 0, None)

    gated_yield    = np.where(productive_cls == 1, yield_prod, 0.0)
    return expected_yield, gated_yield, productive_prob


def optimized_feature_importance(bundle):
    """Weighted average of base-learner importances (spatial lag treated separately)."""
    all_features = bundle['all_features']
    all_feat_ext = all_features + [
        'district_log_median',
        'district_log_iqr',
        'spatial_lag_yield',
        'log_nearest_dist'
    ]
    base_models = bundle['stage2']['base_models']
    n_base_feats = len(all_feat_ext)

    importances = np.zeros(n_base_feats)
    for m in base_models:
        if hasattr(m, 'feature_importances_'):
            imp = m.feature_importances_
            if len(imp) == n_base_feats:
                importances += imp / len(base_models)
            else:
                importances[:len(imp)] += imp / len(base_models)

    return pd.DataFrame({
        'Feature': all_feat_ext[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False)


def robust_site_yield(series):
    vals = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return 0.0
    productive_vals = vals[vals > 0]
    if productive_vals.size == 0:
        return 0.0
    # Mixed dry/productive repeats at the same coordinates are common in field logs.
    # Keep the productive magnitude signal while removing contradictory duplicate labels.
    return float(np.median(productive_vals))


def deduplicate_boreholes_for_modeling(boreholes, coord_precision=4):
    working = boreholes.copy()
    working['lat_round'] = pd.to_numeric(working['lat'], errors='coerce').round(coord_precision)
    working['lon_round'] = pd.to_numeric(working['lon'], errors='coerce').round(coord_precision)
    working = working.dropna(subset=['lat_round', 'lon_round'])

    group_cols = ['lat_round', 'lon_round']
    grouped = working.groupby(group_cols, sort=False)

    dedup = grouped.first().reset_index()
    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in group_cols]
    object_cols = [c for c in working.columns if c not in numeric_cols + group_cols]

    for col in numeric_cols:
        if col == 'yield':
            dedup[col] = grouped[col].apply(robust_site_yield).to_numpy()
        else:
            dedup[col] = grouped[col].median().to_numpy()

    for col in object_cols:
        dedup[col] = grouped[col].apply(mode_or_first).to_numpy()

    conflict_tbl = grouped['yield'].agg(['count', 'min', 'max']).reset_index()
    mixed_sites = ((conflict_tbl['min'] <= 0) & (conflict_tbl['max'] > 0)).sum()
    stats = {
        'total_records': int(len(working)),
        'unique_sites': int(len(dedup)),
        'collapsed_duplicates': int(len(working) - len(dedup)),
        'mixed_dry_productive_sites': int(mixed_sites)
    }

    dedup = dedup.drop(columns=['lat_round', 'lon_round'], errors='ignore')
    return dedup, stats


def robust_log_target(y, lower_q=0.03, upper_q=0.97):
    y_log = np.log1p(np.clip(np.asarray(y, dtype=float), 0, None))
    if y_log.size < 10:
        return y_log
    lo, hi = np.quantile(y_log, [lower_q, upper_q])
    return np.clip(y_log, lo, hi)


def safe_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}
    r2 = r2_score(y_true, y_pred) if y_true.size >= 2 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def fit_stage2_target_transform(y_train, upper_quantile=0.95):
    y_train = np.asarray(y_train, dtype=float)
    positive = y_train[y_train > 0]
    if len(positive) == 0:
        clip_upper = 1.0
    else:
        clip_upper = float(np.nanquantile(positive, upper_quantile))
        if not np.isfinite(clip_upper) or clip_upper <= 0:
            clip_upper = float(np.nanmax(positive))
        clip_upper = max(clip_upper, 1.0)
    return {'clip_upper': clip_upper}


def transform_stage2_target(y, params):
    y = np.asarray(y, dtype=float)
    clipped = np.clip(y, 0, params['clip_upper'])
    return np.log1p(clipped)


def inverse_stage2_target(y_log, params):
    y = np.expm1(np.asarray(y_log, dtype=float))
    return np.clip(y, 0, params['clip_upper'])
    
def fit_stage2_linear_calibrator(y_true, y_pred):
    """Fit conservative affine calibration y_true ~= a * y_pred + b using OOF predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 8 or len(y_pred) < 8:
        return {'slope': 1.0, 'intercept': 0.0}

    x_std = float(np.nanstd(y_pred))
    if not np.isfinite(x_std) or x_std < 1e-9:
        return {'slope': 1.0, 'intercept': 0.0}

    try:
        slope, intercept = np.polyfit(y_pred, y_true, deg=1)
        if not np.isfinite(slope):
            slope = 1.0
        if not np.isfinite(intercept):
            intercept = 0.0
        slope = float(np.clip(slope, 0.6, 1.4))
        intercept = float(np.clip(intercept, -0.6, 0.6))
        return {'slope': slope, 'intercept': intercept}
    except Exception:
        return {'slope': 1.0, 'intercept': 0.0}

def apply_stage2_linear_calibrator(y_pred, calibrator):
    y_pred = np.asarray(y_pred, dtype=float)
    slope = float(calibrator.get('slope', 1.0))
    intercept = float(calibrator.get('intercept', 0.0))
    return slope * y_pred + intercept


def fit_stage2_residual_corrector(X_resid, y_resid):
    """Fit a conservative residual corrector on OOF stage-2 errors."""
    X_resid = np.asarray(X_resid, dtype=float)
    y_resid = np.asarray(y_resid, dtype=float)
    if len(y_resid) < 10:
        return None
    try:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])
        model.fit(X_resid, y_resid)
        return model
    except Exception:
        return None


def apply_stage2_residual_corrector(y_pred, model, X_resid):
    if model is None:
        return np.asarray(y_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    X_resid = np.asarray(X_resid, dtype=float)
    try:
        correction = model.predict(X_resid)
        correction = np.clip(correction, -0.8, 0.8)
        return y_pred + correction
    except Exception:
        return y_pred


def fit_stage2_isotonic_calibrator(y_true, y_pred):
    """Fit monotonic post-calibration to improve zone-level calibration without breaking rank order."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 12:
        return None
    if np.nanstd(y_pred) < 1e-8:
        return None
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_pred, y_true)
        return iso
    except Exception:
        return None


def apply_stage2_isotonic_calibrator(y_pred, calibrator):
    y_pred = np.asarray(y_pred, dtype=float)
    if calibrator is None:
        return y_pred
    try:
        return np.asarray(calibrator.predict(y_pred), dtype=float)
    except Exception:
        return y_pred


def crossfit_stage2_isotonic_predictions(y_true, y_pred, groups):
    """Leakage-safe isotonic calibration via group-wise cross-fitting on OOF predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)

    if len(y_true) < 12:
        return y_pred

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return y_pred

    n_splits = max(2, min(5, len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    calibrated = y_pred.copy()

    for tr_idx, va_idx in gkf.split(y_pred, y_true, groups):
        iso = fit_stage2_isotonic_calibrator(y_true[tr_idx], y_pred[tr_idx])
        if iso is None:
            calibrated[va_idx] = y_pred[va_idx]
        else:
            calibrated[va_idx] = apply_stage2_isotonic_calibrator(y_pred[va_idx], iso)

    return calibrated


def compute_stage2_district_metrics(y_true, y_pred, districts):
    from sklearn.metrics import median_absolute_error as _medae
    eval_df = pd.DataFrame({
        'district_new': districts,
        'y_true': y_true,
        'y_pred': y_pred
    })
    rows = []
    for district, part in eval_df.groupby('district_new'):
        n = len(part)
        if n == 0:
            continue
        row = {
            'district_new': district,
            'n': int(n),
            'rmse': float(np.sqrt(mean_squared_error(part['y_true'], part['y_pred']))),
            'mae': float(mean_absolute_error(part['y_true'], part['y_pred'])),
            'medae': float(_medae(part['y_true'], part['y_pred'])),
            'r2': float(r2_score(part['y_true'], part['y_pred'])) if n >= 2 else np.nan
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['district_new', 'n', 'rmse', 'mae', 'medae', 'r2'])
    return pd.DataFrame(rows).sort_values(['n', 'rmse'], ascending=[False, True])


class SpatialRidgeMetaLearner:
    def __init__(self, alpha=1.0, gamma=1.0, k_neighbors=5):
        self.alpha = alpha
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.coef_ = None

    def fit(self, X_meta, y, coords):
        X_meta = np.asarray(X_meta, dtype=float)
        y = np.asarray(y, dtype=float)
        coords = np.asarray(coords, dtype=float)

        L = build_laplacian_from_coords(coords, k_neighbors=self.k_neighbors)
        p = X_meta.shape[1]
        ridge = self.alpha * np.eye(p)
        spatial = self.gamma * (X_meta.T @ L @ X_meta)
        A = X_meta.T @ X_meta + ridge + spatial
        b = X_meta.T @ y
        self.coef_ = np.linalg.pinv(A) @ b
        return self

    def predict(self, X_meta):
        return np.asarray(X_meta, dtype=float) @ self.coef_


def fit_stage2_models(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb = XGBRegressor(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.3,
        min_child_weight=5,
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return rf, xgb


def create_stage1_confusion_matrix(y_true, y_pred, output_path="results/stage1_confusion_matrix.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Stage 1 Confusion Matrix (Dry vs Productive)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_stage2_obs_pred_plot(y_true, y_pred, output_path="results/stage2_observed_vs_predicted.png"):
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor='black', linewidth=0.3)
    lim_min = min(np.min(y_true), np.min(y_pred))
    lim_max = max(np.max(y_true), np.max(y_pred))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2)
    plt.xlabel('Observed Yield (m³/h)')
    plt.ylabel('Predicted Yield (m³/h)')
    plt.title('Stage 2 Spatial CV: Observed vs Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_manuscript_training(boreholes, features, use_spatial_cv=True, report_publishable_metrics=False):
    boreholes_model, dedupe_stats = deduplicate_boreholes_for_modeling(boreholes)

    X_all_raw = boreholes_model[features].copy()
    y_all = boreholes_model['yield'].to_numpy(dtype=float)
    y_class = (y_all > 0).astype(int)
    coords_all = boreholes_model[['lon', 'lat']].to_numpy(dtype=float)
    groups_all = build_spatial_groups(boreholes_model)
    district_labels = boreholes_model['district_new'].fillna('unknown').astype(str) if 'district_new' in boreholes_model.columns else pd.Series(groups_all.astype(str))

    if use_spatial_cv:
        unique_groups = np.unique(groups_all)
        n_splits = max(2, min(5, len(unique_groups)))
        outer_cv = GroupKFold(n_splits=n_splits)
        outer_split_iter = outer_cv.split(X_all_raw, y_class, groups_all)
    else:
        n_splits = max(2, min(5, len(boreholes_model)))
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        outer_split_iter = outer_cv.split(X_all_raw, y_class)

    stage1_oof = np.zeros(len(boreholes_model), dtype=int)
    stage1_prob = np.zeros(len(boreholes_model), dtype=float)
    stage2_oof_log = np.full(len(boreholes_model), np.nan, dtype=float)

    fold_rows = []
    district_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(outer_split_iter, start=1):
        X_train_raw = X_all_raw.iloc[train_idx]
        X_val_raw = X_all_raw.iloc[val_idx]
        y_train_cls = y_class[train_idx]

        imputer1 = KNNImputer(n_neighbors=3)
        scaler1 = StandardScaler()
        X_train_1 = scaler1.fit_transform(imputer1.fit_transform(X_train_raw))
        X_val_1 = scaler1.transform(imputer1.transform(X_val_raw))

        cls = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        cls.fit(X_train_1, y_train_cls)
        stage1_oof[val_idx] = cls.predict(X_val_1)
        stage1_prob[val_idx] = cls.predict_proba(X_val_1)[:, 1]

        productive_train_mask = y_train_cls == 1
        productive_val_mask = y_class[val_idx] == 1
        if productive_train_mask.sum() < 8 or productive_val_mask.sum() == 0:
            continue

        train_prod_idx = np.array(train_idx)[productive_train_mask]
        val_prod_idx = np.array(val_idx)[productive_val_mask]

        X_train_prod_raw = X_all_raw.iloc[train_prod_idx]
        X_val_prod_raw = X_all_raw.iloc[val_prod_idx]
        y_train_prod_log = robust_log_target(y_all[train_prod_idx])

        imputer2 = KNNImputer(n_neighbors=3)
        scaler2 = StandardScaler()
        X_train_prod = scaler2.fit_transform(imputer2.fit_transform(X_train_prod_raw))
        X_val_prod = scaler2.transform(imputer2.transform(X_val_prod_raw))

        groups_train_prod = groups_all[train_prod_idx]
        coords_train_prod = coords_all[train_prod_idx]
        if use_spatial_cv:
            unique_inner = np.unique(groups_train_prod)
            inner_splits = max(2, min(4, len(unique_inner)))
            inner_splitter = GroupKFold(n_splits=inner_splits)
            inner_iter = inner_splitter.split(X_train_prod, y_train_prod_log, groups_train_prod)
        else:
            inner_splits = max(2, min(4, len(train_prod_idx)))
            inner_splitter = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
            inner_iter = inner_splitter.split(X_train_prod, y_train_prod_log)

        base_oof = np.zeros((len(train_prod_idx), 2), dtype=float)
        for i_tr, i_va in inner_iter:
            rf_i, xgb_i = fit_stage2_models(X_train_prod[i_tr], y_train_prod_log[i_tr])
            base_oof[i_va, 0] = rf_i.predict(X_train_prod[i_va])
            base_oof[i_va, 1] = xgb_i.predict(X_train_prod[i_va])

        meta = SpatialRidgeMetaLearner(alpha=1.0, gamma=1.0, k_neighbors=5)
        meta.fit(base_oof, y_train_prod_log, coords_train_prod)

        rf_full, xgb_full = fit_stage2_models(X_train_prod, y_train_prod_log)
        base_val = np.column_stack([
            rf_full.predict(X_val_prod),
            xgb_full.predict(X_val_prod)
        ])
        val_pred_log = meta.predict(base_val)
        stage2_oof_log[val_prod_idx] = val_pred_log

        y_fold_true = y_all[val_prod_idx]
        y_fold_pred = np.expm1(val_pred_log)
        fold_metric = compute_publishable_metrics(y_fold_true, y_fold_pred) if report_publishable_metrics else safe_regression_metrics(y_fold_true, y_fold_pred)
        fold_rows.append({
            'fold': fold_id,
            'n_productive': int(len(val_prod_idx)),
            **fold_metric
        })

        val_districts = district_labels.iloc[val_prod_idx].to_numpy()
        for district in np.unique(val_districts):
            district_mask = val_districts == district
            y_dist_true = y_fold_true[district_mask]
            y_dist_pred = y_fold_pred[district_mask]
            dist_metric = safe_regression_metrics(y_dist_true, y_dist_pred)
            district_rows.append({
                'fold': fold_id,
                'district': district,
                'n_productive': int(district_mask.sum()),
                **dist_metric
            })

    stage1_metrics = {
        'accuracy': accuracy_score(y_class, stage1_oof),
        'precision': precision_score(y_class, stage1_oof, zero_division=0),
        'recall': recall_score(y_class, stage1_oof, zero_division=0),
        'f1': f1_score(y_class, stage1_oof, zero_division=0)
    }

    valid_stage2 = np.isfinite(stage2_oof_log)
    y_stage2_true = y_all[valid_stage2]
    y_stage2_pred = np.expm1(stage2_oof_log[valid_stage2])
    stage2_metrics = compute_publishable_metrics(y_stage2_true, y_stage2_pred) if report_publishable_metrics else safe_regression_metrics(y_stage2_true, y_stage2_pred)

    fold_metrics_df = pd.DataFrame(fold_rows)
    district_metrics_df = pd.DataFrame(district_rows)

    create_stage1_confusion_matrix(y_class, stage1_oof)
    create_stage2_obs_pred_plot(y_stage2_true, y_stage2_pred)

    imputer1_full = KNNImputer(n_neighbors=3)
    scaler1_full = StandardScaler()
    X1_full = scaler1_full.fit_transform(imputer1_full.fit_transform(X_all_raw))
    cls_full = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    cls_full.fit(X1_full, y_class)

    prod_mask_full = y_class == 1
    X_prod_raw = X_all_raw.iloc[prod_mask_full]
    y_prod_log = robust_log_target(y_all[prod_mask_full])
    coords_prod = coords_all[prod_mask_full]
    groups_prod = groups_all[prod_mask_full]

    imputer2_full = KNNImputer(n_neighbors=3)
    scaler2_full = StandardScaler()
    X2_full = scaler2_full.fit_transform(imputer2_full.fit_transform(X_prod_raw))

    unique_outer = np.unique(groups_prod)
    outer_splits = max(2, min(5, len(unique_outer)))
    outer_gkf = GroupKFold(n_splits=outer_splits)
    base_oof_full = np.zeros((len(X2_full), 2), dtype=float)
    for i_tr, i_va in outer_gkf.split(X2_full, y_prod_log, groups_prod):
        rf_i, xgb_i = fit_stage2_models(X2_full[i_tr], y_prod_log[i_tr])
        base_oof_full[i_va, 0] = rf_i.predict(X2_full[i_va])
        base_oof_full[i_va, 1] = xgb_i.predict(X2_full[i_va])

    meta_full = SpatialRidgeMetaLearner(alpha=1.0, gamma=1.0, k_neighbors=5)
    meta_full.fit(base_oof_full, y_prod_log, coords_prod)
    rf_full, xgb_full = fit_stage2_models(X2_full, y_prod_log)

    return {
        'stage1': {
            'imputer': imputer1_full,
            'scaler': scaler1_full,
            'model': cls_full,
            'metrics': stage1_metrics
        },
        'stage2': {
            'imputer': imputer2_full,
            'scaler': scaler2_full,
            'rf': rf_full,
            'xgb': xgb_full,
            'meta': meta_full,
            'metrics': stage2_metrics,
            'train_features': X2_full,
            'train_targets_log': y_prod_log
        },
        'oof': {
            'stage2_true': y_stage2_true,
            'stage2_pred': y_stage2_pred
        },
        'diagnostics': {
            'deduplication': dedupe_stats,
            'stage2_coverage': {
                'n_stage2_valid': int(valid_stage2.sum()),
                'n_total_records': int(len(y_all))
            },
            'fold_metrics': fold_metrics_df,
            'district_metrics': district_metrics_df
        }
    }


def manuscript_predict_yield(bundle, X_raw):
    X_stage1 = bundle['stage1']['scaler'].transform(bundle['stage1']['imputer'].transform(X_raw))
    productive_prob = bundle['stage1']['model'].predict_proba(X_stage1)[:, 1]
    productive_cls = bundle['stage1']['model'].predict(X_stage1)

    X_stage2 = bundle['stage2']['scaler'].transform(bundle['stage2']['imputer'].transform(X_raw))
    base_pred = np.column_stack([
        bundle['stage2']['rf'].predict(X_stage2),
        bundle['stage2']['xgb'].predict(X_stage2)
    ])
    yield_prod = np.expm1(bundle['stage2']['meta'].predict(base_pred))
    yield_prod = np.clip(yield_prod, 0, None)

    expected_yield = productive_prob * yield_prod
    gated_yield = np.where(productive_cls == 1, yield_prod, 0.0)
    return expected_yield, gated_yield, productive_prob


def manuscript_feature_importance(bundle, features):
    rf_imp = bundle['stage2']['rf'].feature_importances_
    xgb_imp = bundle['stage2']['xgb'].feature_importances_
    meta_w = np.abs(bundle['stage2']['meta'].coef_)
    if meta_w.sum() == 0:
        meta_w = np.array([0.5, 0.5])
    else:
        meta_w = meta_w / meta_w.sum()
    combined = rf_imp * meta_w[0] + xgb_imp * meta_w[1]
    return pd.DataFrame({'Feature': features, 'Importance': combined}).sort_values('Importance', ascending=False)

# --- Main Pipeline ---
def main(
    profile_name='publication',
    boreholes_file='Boreholes.csv',
    ssl_latent_dim=4,
    ssl_hidden_dim=16,
    ssl_epochs=120,
    ssl_mask_prob=0.10,
    ssl_lr=1e-3):
    if profile_name not in RUN_PROFILES:
        raise ValueError(f"Unknown profile '{profile_name}'. Valid options: {list(RUN_PROFILES.keys())}")
    profile = RUN_PROFILES[profile_name]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading Central Region shapefile...")
    shapefile_path = "central_region_ghana.shp"
    central_region_gdf = load_shapefile(shapefile_path)
    region_bounds = get_shapefile_bounds(central_region_gdf)

    print(f"Loading boreholes data from {boreholes_file}...")
    try:
        boreholes = pd.read_csv(boreholes_file, encoding='ISO-8859-1')
        boreholes.columns = boreholes.columns.str.strip().str.lower()

        # Support alternate coordinate column names used in research-grade datasets.
        if 'lat' not in boreholes.columns and 'latitude' in boreholes.columns:
            boreholes['lat'] = boreholes['latitude']
        if 'lon' not in boreholes.columns and 'longitude' in boreholes.columns:
            boreholes['lon'] = boreholes['longitude']
        if 'district_new' not in boreholes.columns and 'district' in boreholes.columns:
            boreholes['district_new'] = boreholes['district']

        if 'lat' not in boreholes.columns or 'lon' not in boreholes.columns:
            raise KeyError("Missing coordinate columns. Expected lat/lon or latitude/longitude")

        boreholes[['lat', 'lon']] = boreholes[['lat', 'lon']].astype(float).round(4)

        yield_column = next((col for col in ['yield', 'pump_test_yield (m3/h)',
                           'yield_m3h', 'yield (m3/h)', 'yield_m3_h',
                           'devpt_yield (m3/h)', 'pump_test_yield (l/min)',
                           'pump_test_yield_lmin']
                           if col in boreholes.columns), None)
        if not yield_column:
            raise KeyError("No yield column found in boreholes file")

        boreholes = boreholes.rename(columns={yield_column: 'yield'})
        if 'l/min' in yield_column or 'lmin' in yield_column:
            boreholes['yield'] = boreholes['yield'] * 0.06

        if boreholes['yield'].isna().any():
            imputer = KNNImputer(n_neighbors=3)
            boreholes[['lat', 'lon', 'yield']] = imputer.fit_transform(boreholes[['lat', 'lon', 'yield']])
        boreholes = boreholes.dropna(subset=['yield'])
        boreholes = filter_points_in_polygon(boreholes, central_region_gdf)

        # Log yield distribution for debugging
        print("Yield distribution:")
        print(boreholes['yield'].describe())
        print(f"Number of unique yields: {len(boreholes['yield'].unique())}")
    except Exception as e:
        print(f"Error loading boreholes data: {e}")
        raise

    if len(boreholes) == 0:
        raise ValueError("No boreholes fall inside the Central Region polygon after spatial filtering")

    print(f"Boreholes inside Central Region: {len(boreholes)}")

    print("Loading external datasets...")
    grace, smap, dem, trmm = load_external_data()
    grace, smap, dem, trmm = filter_external_datasets_to_region(
        [grace, smap, dem, trmm],
        central_region_gdf,
        buffer_deg=profile['external_buffer_deg']
    )
    print(f"External support points - GRACE: {len(grace)}, SMAP: {len(smap)}, DEM: {len(dem)}, Precip: {len(trmm)}")

    global interpolators
    interpolators = {}
    if not grace.empty:
        interpolators['grace_anomaly'] = build_spatial_interpolator(grace, 'grace_anomaly')
    if not smap.empty:
        interpolators['soil_moisture'] = build_spatial_interpolator(smap, 'soil_moisture')
    if not dem.empty:
        interpolators['elevation'] = build_spatial_interpolator(dem, 'elevation')
        interpolators['slope'] = build_spatial_interpolator(dem, 'slope')

    if not trmm.empty and 'precipitation' in trmm.columns:
        print("Using TRMM precipitation data")
        interpolators['precipitation'] = build_spatial_interpolator(trmm, 'precipitation')
    else:
        print("Warning: No valid TRMM precipitation data found")
        if 'precipitation' in boreholes.columns:
            mean_precip = boreholes['precipitation'].mean()
        else:
            mean_precip = trmm['precipitation'].mean() if 'precipitation' in trmm.columns else 0.05
        print(f"Using mean precipitation value: {mean_precip}")
        boreholes['precipitation'] = mean_precip

    boreholes = add_interpolated_features(boreholes)

    print("Creating prediction grid...")
    min_lon = region_bounds['min_lon']
    max_lon = region_bounds['max_lon']
    min_lat = region_bounds['min_lat']
    max_lat = region_bounds['max_lat']

    lat_range = np.arange(min_lat, max_lat, 0.01)
    lon_range = np.arange(min_lon, max_lon, 0.01)
    grid = pd.DataFrame([(lat, lon) for lat in lat_range for lon in lon_range],
                       columns=['lat', 'lon'])
    grid = add_interpolated_features(grid)
    grid = filter_points_in_polygon(grid, central_region_gdf)

    print("Creating geological proxies...")
    required_geo_cols = ['grace_anomaly', 'elevation', 'slope']
    for col in required_geo_cols:
        if col not in boreholes.columns:
            raise ValueError(f"Missing required column in boreholes: {col}")
        if boreholes[col].isna().any():
            print(f"Imputing missing values in boreholes {col}...")
            imputer = KNNImputer(n_neighbors=3)
            boreholes[[col]] = imputer.fit_transform(boreholes[[col]])

    boreholes = create_geological_proxies(boreholes)

    for col in required_geo_cols:
        if col not in grid.columns:
            raise ValueError(f"Missing required column in grid: {col}")
        if grid[col].isna().any():
            print(f"Imputing missing values in grid {col}...")
            imputer = KNNImputer(n_neighbors=3)
            grid[[col]] = imputer.fit_transform(grid[[col]])

    grid = create_geological_proxies(grid)

    print("Engineering features...")
    temporal_lags = profile['temporal_lags']
    boreholes = enhanced_feature_engineering(boreholes, temporal_lags=temporal_lags)
    grid = enhanced_feature_engineering(grid, temporal_lags=temporal_lags)

    print("Preprocessing data...")
    features = [
        'grace_anomaly', 'soil_moisture', 'elevation', 'slope', 'precipitation',
        'sbgi', 'sbgi_geology', 'topo_wetness', 'elev_precip',
        'slope_precip', 'elevation_squared', 'recharge_potential',
        'geo_cluster', 'geo_dist_0', 'geo_dist_1', 'geo_dist_2',
        'high_terrain', 'steep_slope', 'voltaian_proxy', 'birimian_proxy', 'tarkwaian_proxy'
    ]
    for lag in range(1, temporal_lags + 1):
        features.append(f'precipitation_t-{lag}')

    # Select boreholes for map overlay only (not used as holdout validation)
    boreholes_subset = boreholes.sample(frac=0.3, random_state=RANDOM_STATE)

    if profile_name in {'optimized', 'optimized_ssl'}:
        print("Training OPTIMIZED two-stage framework (RF+XGB+LGB / GP-SpatialRidge / SBGWI enhanced)...")

        # Prepare enhanced SBGWI on boreholes before training
        boreholes = compute_enhanced_sbgwi(boreholes)
        extra_sbgwi = ['sbgwi_log', 'sbgwi_precip', 'sbgwi_grace', 'sbgwi_topo', 'sbgwi_geo_mult']
        opt_features = features + [f for f in extra_sbgwi if f not in features]

        # Prepare enhanced SBGWI on grid too
        grid = compute_enhanced_sbgwi(grid)
        # Fill any NaNs introduced in grid
        for f in extra_sbgwi:
            if f in grid.columns:
                grid[f] = grid[f].fillna(0.0)

        train_features = list(features)
        if profile_name == 'optimized_ssl':
            print("Building self-supervised latent features (denoising autoencoder)...")
            ssl_source_features = opt_features
            z_b, z_g, latent_cols = build_ssl_latent_features(
                boreholes,
                grid,
                ssl_source_features,
                latent_dim=ssl_latent_dim,
                hidden_dim=ssl_hidden_dim,
                epochs=ssl_epochs,
                lr=ssl_lr,
                mask_prob=ssl_mask_prob
            )
            if latent_cols:
                for i, col in enumerate(latent_cols):
                    boreholes[col] = z_b[:, i]
                    grid[col] = z_g[:, i]
                train_features = features + latent_cols
                print(f"Added {len(latent_cols)} SSL latent features.")
            else:
                print("SSL latent feature generation skipped (insufficient usable columns).")

        opt_bundle = run_optimized_training(boreholes, train_features)
        all_features = opt_bundle['all_features']

        s1m = opt_bundle['stage1']['metrics']
        s2m = opt_bundle['stage2']['metrics']
        metrics = {
            'Stage1 Accuracy':          f"{s1m['accuracy']:.3f}",
            'Stage1 Precision':         f"{s1m['precision']:.3f}",
            'Stage1 Recall':            f"{s1m['recall']:.3f}",
            'Stage1 F1':                f"{s1m['f1']:.3f}",
            'Stage2 R² (Spatial CV)':   f"{s2m['r2']:.3f}",
            'Stage2 RMSE (Spatial CV)': f"{s2m['rmse']:.3f}",
            'Stage2 MAE (Spatial CV)':  f"{s2m['mae']:.3f}",
            'Stage2 Spearman ρ':        f"{s2m.get('spearman_r', float('nan')):.3f}",
            'Stage2 Zone Accuracy':     f"{s2m.get('zone_accuracy', float('nan')):.3f}",
            'Stage2 Direction Acc.':    f"{s2m.get('direction_accuracy', float('nan')):.3f}",
            'Stage2 Bias (m³/h)':       f"{s2m.get('bias', float('nan')):.3f}",
            'Meta-learner':             str(opt_bundle['stage2']['meta_type']),
            'Base learners':            f"RF + XGBoost{' + LightGBM' if HAS_LIGHTGBM else ''}",
            'SSL Latent Features':      str(sum(1 for f in all_features if str(f).startswith('ssl_latent_'))),
            'SSL latent_dim':           str(ssl_latent_dim),
            'SSL hidden_dim':           str(ssl_hidden_dim),
            'SSL epochs':               str(ssl_epochs),
            'SSL mask_prob':            f"{ssl_mask_prob:.3f}",
            'SSL learning_rate':        f"{ssl_lr:.5f}",
        }
        for name, value in metrics.items():
            print(f"  {name}: {value}")

        feat_imp = optimized_feature_importance(opt_bundle)
        active_profile = 'optimized_ssl' if profile_name == 'optimized_ssl' else 'optimized'
        write_paper_summary(metrics, feat_imp.head(15), active_profile)
        write_reproducibility_manifest(active_profile, metrics, all_features, boreholes_file=boreholes_file)

        diag = opt_bundle.get('diagnostics', {})
        if isinstance(diag.get('fold_metrics'), pd.DataFrame):
            diag['fold_metrics'].to_csv("results/optimized_stage2_fold_metrics.csv", index=False)
        if isinstance(diag.get('district_metrics'), pd.DataFrame):
            diag['district_metrics'].to_csv("results/optimized_stage2_district_metrics.csv", index=False)

        pred_coords = grid[['lon', 'lat']].to_numpy(dtype=float)
        expected_yield, gated_yield, productive_prob = optimized_predict_yield(
            opt_bundle, grid, pred_coords)

        grid['predicted_yield']       = np.clip(expected_yield, 0, 2.5)
        grid['productive_probability'] = productive_prob
        rmse_cv = s2m['rmse']
        grid['predicted_lower'] = np.clip(grid['predicted_yield'] - 1.96 * rmse_cv, 0, 2.5)
        grid['predicted_upper'] = np.clip(grid['predicted_yield'] + 1.96 * rmse_cv, 0, 2.5)

        optimal_breaks = find_optimal_classes(grid['predicted_yield'])
        print("Optimal class breaks:", optimal_breaks)

        safe_save(
            grid[['lat', 'lon', 'predicted_yield', 'predicted_lower',
                  'predicted_upper', 'productive_probability']],
            "results/optimized_predictions.csv"
        )
        # Also overwrite the default results path so map generators pick it up
        safe_save(
            grid[['lat', 'lon', 'predicted_yield', 'predicted_lower',
                  'predicted_upper', 'productive_probability']],
            "results/enhanced_predictions.csv"
        )
        joblib.dump(opt_bundle, "models/optimized_model.pkl")
        feat_imp.to_csv("results/feature_importance.csv", index=False)

        # Meta-learner weight visualization
        meta_obj = opt_bundle['stage2']['meta']
        if hasattr(meta_obj, 'coef_') and meta_obj.coef_ is not None:
            base_names = ['RandomForest', 'XGBoost'] + (['LightGBM'] if HAS_LIGHTGBM else [])
            meta_weights = pd.DataFrame({
                'Model': base_names[:len(meta_obj.coef_)],
                'Weight': np.abs(meta_obj.coef_)
            })
            meta_weights.to_csv("results/meta_model_weights.csv", index=False)
            plt.figure(figsize=(8, 4))
            sns.barplot(x='Weight', y='Model', data=meta_weights)
            plt.title("Stage-2 Spatial-Ridge Meta Weights (Optimized)")
            plt.tight_layout()
            plt.savefig("results/feature_importances.png")
            plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp.head(15), x='Importance', y='Feature')
        plt.title("Top 15 Feature Importances – Optimized Model")
        plt.tight_layout()
        plt.savefig("results/feature_contribution_chart.png", dpi=300, bbox_inches='tight')
        plt.close()

    elif profile_name == 'manuscript':
        print("Training manuscript two-stage framework...")
        manuscript_bundle = run_manuscript_training(boreholes, features)

        stage1_metrics = manuscript_bundle['stage1']['metrics']
        stage2_metrics = manuscript_bundle['stage2']['metrics']
        metrics = {
            'Stage1 Accuracy': f"{stage1_metrics['accuracy']:.3f}",
            'Stage1 Precision': f"{stage1_metrics['precision']:.3f}",
            'Stage1 Recall': f"{stage1_metrics['recall']:.3f}",
            'Stage1 F1': f"{stage1_metrics['f1']:.3f}",
            'Stage2 R² (Spatial CV)': f"{stage2_metrics['r2']:.3f}",
            'Stage2 RMSE (Spatial CV)': f"{stage2_metrics['rmse']:.3f}",
            'Stage2 MAE (Spatial CV)': f"{stage2_metrics['mae']:.3f}"
        }
        for name, value in metrics.items():
            print(f"{name}: {value}")

        feature_importance = manuscript_feature_importance(manuscript_bundle, features)
        write_paper_summary(metrics, feature_importance.head(15), profile_name)
        write_reproducibility_manifest(profile_name, metrics, features, boreholes_file=boreholes_file)

        diag = manuscript_bundle.get('diagnostics', {})
        if 'fold_metrics' in diag and isinstance(diag['fold_metrics'], pd.DataFrame):
            diag['fold_metrics'].to_csv("results/stage2_fold_metrics.csv", index=False)
        if 'district_metrics' in diag and isinstance(diag['district_metrics'], pd.DataFrame):
            diag['district_metrics'].to_csv("results/stage2_district_metrics.csv", index=False)
        if 'deduplication' in diag:
            pd.DataFrame([diag['deduplication']]).to_csv("results/training_deduplication_summary.csv", index=False)

        expected_yield, gated_yield, productive_prob = manuscript_predict_yield(manuscript_bundle, grid[features])
        grid['predicted_yield'] = np.clip(expected_yield, 0, 2.5)
        grid['productive_probability'] = productive_prob

        rmse_cv = stage2_metrics['rmse']
        grid['predicted_lower'] = np.clip(grid['predicted_yield'] - 1.96 * rmse_cv, 0, 2.5)
        grid['predicted_upper'] = np.clip(grid['predicted_yield'] + 1.96 * rmse_cv, 0, 2.5)

        optimal_breaks = find_optimal_classes(grid['predicted_yield'])
        print("Optimal class breaks:", optimal_breaks)

        safe_save(
            grid[['lat', 'lon', 'predicted_yield', 'predicted_lower', 'predicted_upper', 'productive_probability']],
            "results/enhanced_predictions.csv"
        )
        joblib.dump(manuscript_bundle, "models/enhanced_model.pkl")

        meta_weights = pd.DataFrame({
            'Model': ['RandomForest', 'XGBoost'],
            'Weight': np.abs(manuscript_bundle['stage2']['meta'].coef_)
        }).sort_values('Weight', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Weight', y='Model', data=meta_weights)
        plt.title("Stage-2 Spatial-Ridge Meta Weights")
        plt.tight_layout()
        plt.savefig("results/feature_importances.png")
        plt.close()

        feature_importance.to_csv("results/feature_importance.csv", index=False)
        meta_weights.to_csv("results/meta_model_weights.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
        plt.title("Top 15 Stage-2 Feature Importances")
        plt.tight_layout()
        plt.savefig("results/feature_contribution_chart.png", dpi=300, bbox_inches='tight')
        plt.close()

    elif profile_name == 'ranked':
        print("Training rank-first non-spatial two-stage framework...")
        ranked_bundle = run_manuscript_training(
            boreholes,
            features,
            use_spatial_cv=False,
            report_publishable_metrics=True
        )

        stage1_metrics = ranked_bundle['stage1']['metrics']
        stage2_metrics = ranked_bundle['stage2']['metrics']
        metrics = {
            'Stage1 Accuracy': f"{stage1_metrics['accuracy']:.3f}",
            'Stage1 Precision': f"{stage1_metrics['precision']:.3f}",
            'Stage1 Recall': f"{stage1_metrics['recall']:.3f}",
            'Stage1 F1': f"{stage1_metrics['f1']:.3f}",
            'Stage2 R² (Non-spatial CV)': f"{stage2_metrics['r2']:.3f}",
            'Stage2 RMSE (Non-spatial CV)': f"{stage2_metrics['rmse']:.3f}",
            'Stage2 MAE (Non-spatial CV)': f"{stage2_metrics['mae']:.3f}",
            'Stage2 Spearman ρ': f"{stage2_metrics.get('spearman_r', float('nan')):.3f}",
            'Stage2 Zone Accuracy': f"{stage2_metrics.get('zone_accuracy', float('nan')):.3f}",
            'Stage2 Direction Acc.': f"{stage2_metrics.get('direction_accuracy', float('nan')):.3f}",
            'Stage2 Bias (m³/h)': f"{stage2_metrics.get('bias', float('nan')):.3f}"
        }
        for name, value in metrics.items():
            print(f"{name}: {value}")

        feature_importance = manuscript_feature_importance(ranked_bundle, features)
        write_paper_summary(metrics, feature_importance.head(15), 'ranked')
        write_reproducibility_manifest('ranked', metrics, features, boreholes_file=boreholes_file)

        diag = ranked_bundle.get('diagnostics', {})
        if isinstance(diag.get('fold_metrics'), pd.DataFrame):
            diag['fold_metrics'].to_csv("results/ranked_fold_metrics.csv", index=False)
        if isinstance(diag.get('district_metrics'), pd.DataFrame):
            diag['district_metrics'].to_csv("results/ranked_district_metrics.csv", index=False)

        expected_yield, gated_yield, productive_prob = manuscript_predict_yield(ranked_bundle, grid[features])
        grid['predicted_yield'] = np.clip(expected_yield, 0, 2.5)
        grid['productive_probability'] = productive_prob
        rmse_cv = stage2_metrics['rmse']
        grid['predicted_lower'] = np.clip(grid['predicted_yield'] - 1.96 * rmse_cv, 0, 2.5)
        grid['predicted_upper'] = np.clip(grid['predicted_yield'] + 1.96 * rmse_cv, 0, 2.5)

        optimal_breaks = find_optimal_classes(grid['predicted_yield'])
        print("Optimal class breaks:", optimal_breaks)

        safe_save(
            grid[['lat', 'lon', 'predicted_yield', 'predicted_lower', 'predicted_upper', 'productive_probability']],
            "results/ranked_predictions.csv"
        )
        safe_save(
            grid[['lat', 'lon', 'predicted_yield', 'predicted_lower', 'predicted_upper', 'productive_probability']],
            "results/enhanced_predictions.csv"
        )
        joblib.dump(ranked_bundle, "models/ranked_model.pkl")
        feature_importance.to_csv("results/ranked_feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
        plt.title("Top 15 Feature Importances – Rank-First Non-Spatial Model")
        plt.tight_layout()
        plt.savefig("results/ranked_feature_contribution_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        imputer = KNNImputer(n_neighbors=3)
        boreholes[features] = imputer.fit_transform(boreholes[features])
        grid[features] = imputer.transform(grid[features])

        scaler = StandardScaler()
        boreholes[features] = scaler.fit_transform(boreholes[features])
        grid[features] = scaler.transform(grid[features])
        joblib.dump(scaler, "models/scaler.pkl")

        print("Training enhanced ensemble model on full dataset...")
        X = boreholes[features]
        y = np.log1p(boreholes['yield'].clip(0, None))
        model, cv_scores = train_enhanced_model(X, y, boreholes['spatial_group'])

        print("\nModel Evaluation on Full Dataset:")
        y_pred = model.predict(X)
        train_residuals = y - y_pred

        phys_loss = PhysicsInformedLoss(X['elevation'], X['recharge_potential'])
        phys_loss_value = phys_loss(y, y_pred).item()

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred))),
            'MAE': mean_absolute_error(np.expm1(y), np.expm1(y_pred)),
            'R²': r2_score(np.expm1(y), np.expm1(y_pred)),
            'CV MSE': f"{-cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
            'Physical Loss': f"{phys_loss_value:.4f}"
        }

        for name, value in metrics.items():
            print(f"{name}: {value}")

        feature_importance = compute_feature_importance_table(model, features)
        write_paper_summary(metrics, feature_importance.head(15), profile_name)
        write_reproducibility_manifest(profile_name, metrics, features, boreholes_file=boreholes_file)

        print("Generating enhanced predictions...")
        grid_pred = model.predict(grid[features])
        grid['predicted_yield'] = np.expm1(grid_pred)

        lower, upper = calculate_prediction_intervals(model, grid[features], train_residuals)
        grid['predicted_lower'] = np.expm1(lower)
        grid['predicted_upper'] = np.expm1(upper)

        grid['predicted_yield'] = grid['predicted_yield'].clip(0, 2.5)
        grid['predicted_lower'] = grid['predicted_lower'].clip(0, 2.5)
        grid['predicted_upper'] = grid['predicted_upper'].clip(0, 2.5)

        optimal_breaks = find_optimal_classes(grid['predicted_yield'])
        print("Optimal class breaks:", optimal_breaks)

        safe_save(grid[['lat', 'lon', 'predicted_yield', 'predicted_lower', 'predicted_upper']],
                 "results/enhanced_predictions.csv")
        joblib.dump(model, "models/enhanced_model.pkl")

        meta_weights = pd.DataFrame({
            'Model': ['RandomForest', 'XGBoost'],
            'Weight': model.final_estimator_.feature_importances_
        }).sort_values('Weight', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Weight', y='Model', data=meta_weights)
        plt.title("Meta Model Feature Importances")
        plt.tight_layout()
        plt.savefig("results/feature_importances.png")
        plt.close()
        feature_importance.to_csv("results/feature_importance.csv", index=False)
        meta_weights.to_csv("results/meta_model_weights.csv", index=False)

        print("Creating visualizations...")
        create_feature_contribution_chart(model, features)

    create_yield_heatmap(grid, central_region_gdf)

    print("Creating output maps with superimposed validation boreholes...")
    create_static_yield_map(grid, central_region_gdf, boreholes_subset)
    create_geotiff(grid, central_region_gdf)
    create_interactive_map(grid, central_region_gdf, boreholes_subset)

    print("Pipeline complete. Results saved in results/ directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groundwater yield modeling for Central Region Ghana")
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES.keys()),
        default="publication",
        help="Run profile. Use 'paper' for paper-compatible configuration."
    )
    parser.add_argument(
        "--boreholes-file",
        default="Boreholes.csv",
        help="Path to boreholes CSV file (supports schema variants like latitude/longitude, district, yield_m3h)."
    )
    parser.add_argument("--ssl-latent-dim", type=int, default=4,
                        help="Latent dimension for optimized_ssl denoising autoencoder.")
    parser.add_argument("--ssl-hidden-dim", type=int, default=16,
                        help="Hidden layer dimension for optimized_ssl denoising autoencoder.")
    parser.add_argument("--ssl-epochs", type=int, default=120,
                        help="Training epochs for optimized_ssl denoising autoencoder.")
    parser.add_argument("--ssl-mask-prob", type=float, default=0.10,
                        help="Mask probability for denoising corruption in optimized_ssl.")
    parser.add_argument("--ssl-lr", type=float, default=1e-3,
                        help="Learning rate for optimized_ssl denoising autoencoder.")
    args = parser.parse_args()
    main(
        profile_name=args.profile,
        boreholes_file=args.boreholes_file,
        ssl_latent_dim=args.ssl_latent_dim,
        ssl_hidden_dim=args.ssl_hidden_dim,
        ssl_epochs=args.ssl_epochs,
        ssl_mask_prob=args.ssl_mask_prob,
        ssl_lr=args.ssl_lr,
    )