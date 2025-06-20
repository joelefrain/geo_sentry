import ezdxf
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import contextily as cx
import rasterio
from rasterio.transform import from_bounds
import numpy as np

def extract_bbox_from_dxf(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    geometries = []

    for e in msp:
        if e.dxftype() == "LINE":
            start = e.dxf.start
            end = e.dxf.end
            geometries.append(LineString([start[:2], end[:2]]))
        elif e.dxftype() in {"LWPOLYLINE", "POLYLINE"}:
            try:
                points = [tuple(p[:2]) for p in e.get_points()]
                if e.closed:
                    geometries.append(Polygon(points))
                else:
                    geometries.append(LineString(points))
            except Exception:
                continue

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf = gdf.set_crs(epsg=32718)  # Cambia la zona si es necesario
    return gdf.total_bounds, gdf.to_crs(epsg=3857)  # bounding box original y reproyectado a Web Mercator

def download_satellite_image(bounds_utm, bounds_webmercator, output_tif, zoom=17):
    # Obtener imagen satelital del bounding box usando contextily
    minx, miny, maxx, maxy = bounds_webmercator.total_bounds

    # Descargar tiles y combinar en una imagen
    img, ext = cx.bounds2img(minx, miny, maxx, maxy, zoom=zoom, source=cx.providers.Esri.WorldImagery)

    # Transformación raster
    transform = from_bounds(*bounds_utm, img.shape[1], img.shape[0])

    # Guardar como GeoTIFF (RGB)
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype='uint8',
        crs="EPSG:32718",  # Sistema original UTM
        transform=transform
    ) as dst:
        for i in range(3):  # R, G, B
            dst.write(img[:, :, i], i + 1)

    print(f"[✔] Imagen satelital guardada como GeoTIFF en: {output_tif}")

# --- Uso principal ---
if __name__ == "__main__":
    dxf_path = "data/config/sample_client/sample_project/dxf/DME_CHO.dxf"
    tif_path = "data/config/sample_client/sample_project/tif/DME_CHO_satelital.tif"

    bounds_utm, bounds_webmercator = extract_bbox_from_dxf(dxf_path)
    download_satellite_image(bounds_utm, bounds_webmercator, tif_path, zoom=18)
