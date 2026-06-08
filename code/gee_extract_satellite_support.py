
import os

import numpy as np
import pandas as pd

try:
    import ee
except Exception as exc:
    raise ImportError(
        "earthengine-api is required. Install with: pip install earthengine-api"
    ) from exc


def normalize_borehole_schema(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if 'lat' not in df.columns and 'latitude' in df.columns:
        df['lat'] = df['latitude']
    if 'lon' not in df.columns and 'longitude' in df.columns:
        df['lon'] = df['longitude']

    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Boreholes file must contain lat/lon or latitude/longitude columns")

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    return df


def build_support_points(df_points, grid_step_deg=0.08):
    base = df_points[['lat', 'lon']].copy()

    min_lat = base['lat'].min() - 0.12
    max_lat = base['lat'].max() + 0.12
    min_lon = base['lon'].min() - 0.12
    max_lon = base['lon'].max() + 0.12

    grid_lat = np.arange(min_lat, max_lat + 1e-9, grid_step_deg)
    grid_lon = np.arange(min_lon, max_lon + 1e-9, grid_step_deg)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')

    grid = pd.DataFrame({
        'lat': glat.ravel(),
        'lon': glon.ravel()
    })

    all_pts = pd.concat([base, grid], ignore_index=True)
    all_pts = all_pts.round({'lat': 5, 'lon': 5}).drop_duplicates(subset=['lat', 'lon'])
    return all_pts


def points_to_fc(points_df):
    feats = []
    for _, row in points_df.iterrows():
        geom = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
        feats.append(ee.Feature(geom, {'lon': float(row['lon']), 'lat': float(row['lat'])}))
    return ee.FeatureCollection(feats)


def fc_to_dataframe(fc, value_columns):
    info = fc.getInfo()
    rows = []
    for feat in info.get('features', []):
        props = feat.get('properties', {})
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [None, None])
        row = {
            'lon': props.get('lon', coords[0]),
            'lat': props.get('lat', coords[1])
        }
        for col in value_columns:
            row[col] = props.get(col)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out['lat'] = pd.to_numeric(out['lat'], errors='coerce')
        out['lon'] = pd.to_numeric(out['lon'], errors='coerce')
    return out


def sample_band(points_fc, image, band_name, out_name, scale):
    sampled = image.select([band_name]).rename([out_name]).sampleRegions(
        collection=points_fc,
        scale=scale,
        geometries=True
    )
    return fc_to_dataframe(sampled, [out_name])


def choose_collection_mean(collection_ids, start, end, fallback_windows=None):
    """Return mean image from first non-empty collection/window combination."""
    windows = [(start, end)]
    if fallback_windows:
        windows.extend(fallback_windows)

    for cid in collection_ids:
        col = ee.ImageCollection(cid)
        for w_start, w_end in windows:
            test_col = col.filterDate(w_start, w_end)
            try:
                if int(test_col.size().getInfo()) > 0:
                    return test_col.mean(), cid, w_start, w_end
            except Exception:
                continue

    raise RuntimeError(
        f"No imagery found for collections {collection_ids} across windows {windows}"
    )


def ensure_ee_initialized(project=None):
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as init_exc:
        msg = (
            "Failed to initialize Earth Engine. Run `earthengine authenticate` first, "
            "then rerun this script."
        )
        raise RuntimeError(msg) from init_exc


def main():
    parser = argparse.ArgumentParser(description="Extract model-ready satellite support CSVs from Google Earth Engine")
    parser.add_argument('--boreholes-file', default='Boreholes.csv', help='Path to boreholes CSV (used for support points)')
    parser.add_argument('--out-dir', default='.', help='Output directory for GEE_* CSV files')
    parser.add_argument('--start-date', default='2018-01-01', help='Start date for temporal composites')
    parser.add_argument('--end-date', default='2020-12-31', help='End date for temporal composites')
    parser.add_argument('--grid-step-deg', type=float, default=0.08, help='Support grid spacing in degrees')
    parser.add_argument('--ee-project', default=None, help='Optional Google Cloud project for Earth Engine init')
    args = parser.parse_args()

    ensure_ee_initialized(project=args.ee_project)

    boreholes = pd.read_csv(args.boreholes_file, encoding='ISO-8859-1')
    boreholes = normalize_borehole_schema(boreholes)

    support_points = build_support_points(boreholes[['lat', 'lon']], grid_step_deg=args.grid_step_deg)
    fc = points_to_fc(support_points)

    start = args.start_date
    end = args.end_date

    # Precipitation (mm/day) composite.
    precip_img = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start, end).mean()
    precip_df = sample_band(fc, precip_img, 'precipitation', 'precipitation', scale=5500)

    # GRACE / GRACE-FO equivalent water thickness anomaly.
    grace_img, grace_source, grace_start, grace_end = choose_collection_mean(
        ['NASA/GRACE-FO/MASS_GRIDS_V04/LAND', 'NASA/GRACE/MASS_GRIDS_V04/LAND'],
        start,
        end,
        fallback_windows=[('2018-06-01', '2023-12-31'), ('2003-01-01', '2017-06-30')]
    )
    grace_df = sample_band(fc, grace_img, 'lwe_thickness_csr', 'grace_anomaly', scale=25000)

    # SMAP surface soil moisture.
    smap_img = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture').filterDate(start, end).mean()
    smap_df = sample_band(fc, smap_img, 'ssm', 'soil_moisture', scale=10000)

    # DEM + slope.
    dem_img = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope_img = ee.Terrain.slope(dem_img).rename('slope')
    dem_stack = dem_img.rename('elevation').addBands(slope_img)
    dem_sampled = dem_stack.sampleRegions(collection=fc, scale=90, geometries=True)
    dem_df = fc_to_dataframe(dem_sampled, ['elevation', 'slope'])

    os.makedirs(args.out_dir, exist_ok=True)

    out_precip = os.path.join(args.out_dir, 'GEE_GPM_IMERG.csv')
    out_grace = os.path.join(args.out_dir, 'GEE_GRACE_FO.csv')
    out_smap = os.path.join(args.out_dir, 'GEE_SMAP_SOIL_MOISTURE.csv')
    out_dem = os.path.join(args.out_dir, 'GEE_DEM.csv')

    precip_df.dropna(subset=['lat', 'lon', 'precipitation']).to_csv(out_precip, index=False)
    grace_df.dropna(subset=['lat', 'lon', 'grace_anomaly']).to_csv(out_grace, index=False)
    smap_df.dropna(subset=['lat', 'lon', 'soil_moisture']).to_csv(out_smap, index=False)
    dem_df.dropna(subset=['lat', 'lon', 'elevation', 'slope']).to_csv(out_dem, index=False)

    print(f"Wrote {out_precip}")
    print(f"Wrote {out_grace}")
    print(f"Wrote {out_smap}")
    print(f"Wrote {out_dem}")
    print(f"GRACE source window: {grace_source} [{grace_start} to {grace_end}]")
    print(f"Support points sampled: {len(support_points)}")


if __name__ == '__main__':
    main()
