import os
import time
import ezdxf
import folium
import rasterio
import numpy as np
import geopandas as gpd
from PIL import Image
from shapely.geometry import LineString
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from rasterio.transform import from_bounds


def extract_bounds_latlon(dxf_path, utm_epsg=32718):
    """Extrae el bounding box del DXF y lo convierte a lat/lon (EPSG:4326)."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    geometries = []

    for e in msp:
        if e.dxftype() == "LINE":
            geometries.append(LineString([e.dxf.start[:2], e.dxf.end[:2]]))
        elif e.dxftype() in {"POLYLINE", "LWPOLYLINE"}:
            try:
                points = [tuple(p[:2]) for p in e.get_points()]
                geometries.append(LineString(points))
            except Exception:
                continue

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=f"EPSG:{utm_epsg}")
    gdf_latlon = gdf.to_crs(epsg=4326)
    return gdf_latlon.total_bounds  # [minlon, minlat, maxlon, maxlat]


def create_folium_map(bounds, output_html):
    """Genera un archivo HTML con el mapa satelital y una caja delimitadora."""
    minlon, minlat, maxlon, maxlat = bounds
    center = [(minlat + maxlat) / 2, (minlon + maxlon) / 2]

    m = folium.Map(location=center, zoom_start=17, tiles="Esri.WorldImagery")
    folium.Rectangle(
        bounds=[[minlat, minlon], [maxlat, maxlon]], color="red", weight=2, fill=False
    ).add_to(m)
    m.fit_bounds([[minlat, minlon], [maxlat, maxlon]])
    m.save(output_html)
    print(f"[✔] Mapa HTML generado: {output_html}")


def render_html_to_png(html_path, output_png, width=1024, height=768, delay=5):
    """Captura una imagen PNG del HTML generado usando Selenium."""
    options = Options()
    options.headless = True
    options.add_argument(f"--window-size={width},{height}")
    driver = webdriver.Chrome(options=options)

    driver.get("file://" + os.path.abspath(html_path))
    time.sleep(delay)  # Espera para cargar los tiles

    screenshot = driver.get_screenshot_as_png()
    with open(output_png, "wb") as f:
        f.write(screenshot)

    driver.quit()
    print(f"[✔] Imagen PNG capturada: {output_png}")


def png_to_geotiff(png_path, bounds, output_tif):
    """Convierte la imagen PNG a GeoTIFF con georreferenciación."""
    img = Image.open(png_path).convert("RGB")
    img_np = np.array(img)

    minlon, minlat, maxlon, maxlat = bounds
    height, width = img_np.shape[0], img_np.shape[1]
    transform = from_bounds(minlon, minlat, maxlon, maxlat, width, height)

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img_np.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img_np[:, :, i], i + 1)

    print(f"[✔] GeoTIFF generado: {output_tif}")


def dxf_to_satellite_geotiff(dxf_path, output_basename, utm_zone=18):
    """Pipeline completo: DXF → Imagen satelital PNG → GeoTIFF georreferenciado."""
    utm_epsg = 32700 + utm_zone
    bounds = extract_bounds_latlon(dxf_path, utm_epsg=utm_epsg)

    html_path = output_basename + ".html"
    png_path = output_basename + ".png"
    tif_path = output_basename + ".tif"

    create_folium_map(bounds, html_path)
    render_html_to_png(html_path, png_path)
    png_to_geotiff(png_path, bounds, tif_path)


# --- USO ---
if __name__ == "__main__":
    dxf_input = "data/config/sample_client/sample_project/dxf/PAD_2B_2C.dxf"
    output_base = "data/config/sample_client/sample_project/tif/PAD_2B_2C"
    dxf_to_satellite_geotiff(dxf_input, output_base, utm_zone=17)
