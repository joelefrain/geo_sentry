from enum import Enum
from typing import Dict, Optional, Any, List
import folium
from folium import plugins
import pandas as pd
import utm
from pathlib import Path
import geopandas as gpd

class ButtonType(Enum):
    CHART = ('btn-chart', '#4CAF50', '#45a049', 'Ver Gráfico')
    DATA = ('btn-data', '#2196F3', '#1976D2', 'Ver Data') 
    PROFILE = ('btn-profile', '#FF5722', '#E64A19', 'Ver Perfil')
    
    def __init__(self, class_name: str, color: str, hover_color: str, text: str):
        self.class_name = class_name
        self.color = color
        self.hover_color = hover_color
        self.text = text

class MapStyles:
    SENSOR_STYLES = {
        'temperature': {'color': 'red', 'icon': 'thermometer'},
        'pressure': {'color': 'blue', 'icon': 'dashboard'},
        'level': {'color': 'green', 'icon': 'tint'},
        'default': {'color': 'gray', 'icon': 'info-sign'}
    }

    @staticmethod
    def get_button_css() -> str:
        css = """
            .custom-btn {
                border: none;
                color: white;
                padding: 4px 8px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 12px;
                margin: 2px;
                cursor: pointer;
                border-radius: 3px;
                transition: 0.3s;
            }
        """
        for btn_type in ButtonType:
            css += f"""
                .{btn_type.class_name} {{ background-color: {btn_type.color}; }}
                .{btn_type.class_name}:hover {{ background-color: {btn_type.hover_color}; }}
            """
        return css

    @staticmethod
    def get_modal_css() -> str:
        return """
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.4);
            }
            .modal-content {
                background-color: #fefefe;
                margin: 5% auto;
                padding: 20px;
                border: 1px solid #888;
                width: 80%;
                height: 80%;
                position: relative;
            }
            .close-btn {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
        """

class MapPlotter:
    """Interactive map plotter with support for points and geodata visualization"""
    
    def __init__(self):
        self.map = None
        
    def _add_styles(self):
        css = f"""
            <style>
            {MapStyles.get_button_css()}
            {MapStyles.get_modal_css()}
            </style>
        """
        self.map.get_root().html.add_child(folium.Element(css))
        
    def _add_modal_js(self):
        js = """
            <script>
            function showModal(content) {
                var modal = document.createElement('div');
                modal.className = 'modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <span class="close-btn" onclick="this.parentElement.parentElement.remove()">&times;</span>
                        <iframe src="${content}" style="width:100%;height:calc(100% - 30px);border:none;"></iframe>
                    </div>
                `;
                document.body.appendChild(modal);
                modal.style.display = "block";
                window.onclick = event => event.target == modal && modal.remove();
            }
            </script>
        """
        self.map.get_root().html.add_child(folium.Element(js))

    def _get_file_url(self, filepath: str) -> str:
        """Convert local file path to file:/// URL"""
        return 'file:///' + str(Path(filepath).resolve()).replace('\\', '/')
    
    def _create_button(self, btn_type: ButtonType, url: str) -> str:
        """Generate HTML for a button with specific type and URL"""
        return f"""
            <button class="custom-btn {btn_type.class_name}" 
                    onclick="showModal('{url}')">
                {btn_type.text}
            </button>
        """
    
    def create_map(self, center_lat: float, center_lon: float, **kwargs):
        """Initialize map with satellite imagery and controls"""
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=kwargs.get('zoom_start', 15),
            tiles=kwargs.get('tiles', 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'),
            attr=kwargs.get('attr', 'Esri')
        )
        
        # Add controls
        plugins.Fullscreen().add_to(self.map)
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            primary_area_unit='sqmeters'
        ).add_to(self.map)
        plugins.MousePosition(
            position='bottomleft',
            separator=' | ',
            num_digits=3,
            formatter="function(num) {return L.Util.formatNum(num, 3) + ' ';};"
        ).add_to(self.map)
        plugins.Draw(
            position='topleft',
            draw_options={'marker': False, 'circlemarker': False}
        ).add_to(self.map)
        
        self._add_styles()
        self._add_modal_js()
        
    def plot_points(self, df: pd.DataFrame, chart_paths: Optional[Dict] = None, data_paths: Optional[Dict] = None):
        """Plot sensor points with popup info and interactive buttons"""
        if self.map is None:
            raise ValueError("Create map first using create_map method")
        
        charts = {k: self._get_file_url(v) for k, v in (chart_paths or {}).items()}
        data = {k: self._get_file_url(v) for k, v in (data_paths or {}).items()}
            
        for _, row in df.iterrows():
            style = MapStyles.SENSOR_STYLES.get(row['sensor_type'].lower(), 
                                              MapStyles.SENSOR_STYLES['default'])
            
            buttons = ""
            if row['name'] in charts:
                buttons += self._create_button(ButtonType.CHART, charts[row['name']])
            if row['name'] in data:
                buttons += self._create_button(ButtonType.DATA, data[row['name']])
            
            popup_html = f"""
                <div style="width:200px">
                    <b>{row['name']}</b><br>
                    {row['name_attr']}: {row['value_attr']}<br>
                    Estatus: {row['status']}<br>
                    <div style="display:flex;gap:4px;">
                        {buttons}
                    </div>
                </div>
            """
            
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa'),
            ).add_to(self.map)
    
    def plot_geodata(self, gdf: gpd.GeoDataFrame, name: str, show_profile: bool = False, 
                    profile_path: Optional[str] = None, **kwargs):
        """Plot geodata with optional profile button"""
        if self.map is None:
            raise ValueError("Create map first using create_map method")
        
        style_function = kwargs.get('style_function', lambda x: {})
        geojson = folium.GeoJson(gdf, style_function=style_function)
        
        popup_html = f"<div style='width:200px'><b>{name}</b><br>"
        if show_profile and profile_path:
            popup_html += self._create_button(ButtonType.PROFILE, self._get_file_url(profile_path))
        popup_html += "</div>"
        
        folium.Popup(popup_html, max_width=300).add_to(geojson)
        geojson.add_to(self.map)
    
    def save_map(self, filepath: str):
        """Save map to HTML file"""
        if self.map is None:
            raise ValueError("Create map first using create_map method")
        self.map.save(filepath)

if __name__ == '__main__':
    # Create sample data for Peru (UTM Zone 18S)
    test_data = pd.DataFrame({
        'name': ['Sensor Lima', 'Sensor Cusco', 'Sensor Arequipa'],
        'sensor_type': ['temperature', 'pressure', 'level'],
        'east': [279033, 177369, 246783],  # UTM Easting
        'north': [8665693, 8503382, 8184452],  # UTM Northing
        'zone_number': [18, 18, 18],
        'zone_letter': ['M', 'M', 'M'],
        'name_attr': ['Temperatura', 'Presión', 'Nivel'],
        'value_attr': ['24°C', '1013 hPa', '120 m'],
        'status': ['Activo', 'Inactivo', 'Mantenimiento']
    })

    # Convert UTM to lat/lon
    test_data['lat'], test_data['lon'] = zip(*test_data.apply(
        lambda row: utm.to_latlon(row['east'], row['north'], row['zone_number'], row['zone_letter']), axis=1))

    # Create map instance for Peru (Zone 18S)
    plotter = MapPlotter()
    
    # Center map in Lima coordinates
    plotter.create_map(center_lat=test_data['lat'][0], center_lon=test_data['lon'][0])
    
    # Ejemplo de uso con rutas HTML para gráficos y datos
    chart_paths = {
        'Sensor Lima': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\multi_dataframe_timeseries.html',
        'Sensor Cusco': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\multi_timeseries.html',
        'Sensor Arequipa': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\timeseries.html'
    }
    
    data_paths = {
        'Sensor Lima': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\multi_dataframe_timeseries.html',
        'Sensor Cusco': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\multi_timeseries.html',
        'Sensor Arequipa': r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\timeseries.html'
    }
    
    profile_path = r'C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\timeseries.html'
    
    # Plot test points with both paths
    plotter.plot_points(test_data, chart_paths, data_paths)
    
    # Convert shapefile, GeoJSON, and XML to GeoDataFrame
    gdf_shapefile = gpd.read_file(r"C:\Users\Joel Efraín\Desktop\SHA_LINE.shp")
    gdf_geojson = gpd.read_file(r"C:\Users\Joel Efraín\Desktop\SHA_LINE.geojson")
    
    # Plot geodata with profile button - note the added name parameter
    plotter.plot_geodata(gdf_shapefile, name='Shapefile Line', show_profile=True, profile_path=profile_path, style_function=lambda x: {'color': 'blue'})
    plotter.plot_geodata(gdf_geojson, name='GeoJSON Line', show_profile=True, profile_path=profile_path, style_function=lambda x: {'color': 'red'})
    
    # Save the map
    plotter.save_map('peru_sensors_map.html')
