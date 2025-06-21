import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import ezdxf

from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, Label
from bokeh.models.glyphs import Scatter

from libs.utils.config_variables import SENSOR_VISUAL_CONFIG

class SectionPlotter:
    def __init__(self, line_start_utm, line_end_utm, base_elevation, dxf_path):
        """
        Inicializa el plotter de sección.
        
        :param line_start_utm: Tupla (este, norte) UTM del punto inicial de la recta
        :param line_end_utm: Tupla (este, norte) UTM del punto final de la recta
        :param base_elevation: Elevación base del plano (metros)
        :param dxf_path: Ruta al archivo DXF con el perfil de sección
        """
        self.line_start = line_start_utm
        self.line_end = line_end_utm
        self.base_elevation = base_elevation
        self.dxf_path = dxf_path
        
        # Procesar DXF
        self.terrain_x, self.terrain_y = self._parse_dxf()
        self.line_length = self._calculate_line_length()
        
    def _parse_dxf(self):
        """Extrae los puntos del perfil del terreno del DXF."""
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()
        
        all_points = []
        lines_data = []
        
        # Extraer todas las líneas, polylines, hatch y texto
        for entity in msp.query('LINE LWPOLYLINE HATCH TEXT'):
            if entity.dxftype() == 'LINE':
                points = [(entity.dxf.start.x, entity.dxf.start.y),
                         (entity.dxf.end.x, entity.dxf.end.y)]
                color = entity.dxf.color
            if entity.dxftype() == 'LWPOLYLINE':
                points = [(vertex[0], vertex[1]) for vertex in entity.vertices()]
                color = entity.dxf.color
            if entity.dxftype() == 'HATCH':
                # Para HATCH, no hay puntos directos, pero podemos usar el bounding box
                bbox = entity.bounding_box()
                points = [(bbox.extmin.x, bbox.extmin.y), (bbox.extmax.x, bbox.extmax.y)]
                color = entity.dxf.color
            if entity.dxftype() == 'TEXT':
                # Para TEXT, usamos la posición como un punto
                points = [(entity.dxf.insert.x, entity.dxf.insert.y)]
                color = entity.dxf.color
            
            all_points.extend(points)
            lines_data.append({
                'points': points,
                'color': color
            })
        
        if not lines_data:
            raise ValueError("No se encontraron entidades válidas en el DXF")
        
        # Guardar todas las líneas para dibujarlas después
        self.lines_data = lines_data
        
        # Retornar el rango completo de X
        x_coords = [p[0] for p in all_points]
        return [min(x_coords), max(x_coords)], [0, 0]  # Ya no necesitamos terrain_y
    
    def _calculate_line_length(self):
        """Calcula la longitud de la recta en metros."""
        dx = self.line_end[0] - self.line_start[0]
        dy = self.line_end[1] - self.line_start[1]
        return np.sqrt(dx**2 + dy**2)
    
    def _project_point(self, east, north):
        """Proyecta un punto UTM a la recta y devuelve la posición en la sección."""
        # Convertir a arrays numpy para cálculos vectoriales
        a = np.array(self.line_start)
        b = np.array(self.line_end)
        p = np.array([east, north])
        
        vector_ab = b - a
        vector_ap = p - a
        
        t = np.dot(vector_ap, vector_ab) / np.dot(vector_ab, vector_ab)
        t = np.clip(t, 0, 1)
        
        projected_point = a + t * vector_ab
        x_section = np.linalg.norm(projected_point - a)
        return x_section
    
    def _dxf_color_to_hex(self, color_number):
        """Convierte el número de color DXF a código hexadecimal."""
        # Mapa de colores DXF básicos
        dxf_colors = {
            1: '#FF0000',   # Red
            2: '#FFFF00',   # Yellow
            3: '#00FF00',   # Green
            4: '#00FFFF',   # Cyan
            5: '#0000FF',   # Blue
            6: '#FF00FF',   # Magenta
            7: '#000000',   # Black
            8: '#808080',   # Gray
            9: '#C0C0C0',   # Light Gray
        }
        return dxf_colors.get(color_number, '#000000')  # Negro por defecto

    def plot_section(self, sensors_df, output_path):
        """Genera el gráfico HTML con la sección y sensores."""
        # Configurar figura
        p = figure(title="Sección longitudinal",
                  x_axis_label='Distancia a lo largo de la sección (m)',
                  y_axis_label='Elevación (m)',
                  tools="pan,wheel_zoom,box_zoom,reset,save")

        # Dibujar todas las líneas del DXF
        for line in self.lines_data:
            points = line['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            color = self._dxf_color_to_hex(line['color'])
            p.line(x_coords, y_coords, line_width=2, color=color)

        # Procesar cada sensor
        for _, sensor in sensors_df.iterrows():
            x_section = self._project_point(sensor['east'], sensor['north'])
            y_section = sensor['elevation']

            # Obtener configuración visual del diccionario global
            config = SENSOR_VISUAL_CONFIG[sensor['sensor_type']]

            # Crear glyph para el sensor
            source = ColumnDataSource(data={
                'x': [x_section],
                'y': [y_section],
                'name': [sensor['name']],
                'value': [sensor['value_attr']]
            })
            
            # Agregar punto del sensor
            glyph = Scatter(x='x', y='y',
                           marker=config['marker'],
                           size=12,
                           fill_color=config['color'],
                           line_color='black')
            
            p.add_glyph(source, glyph)
            
            # Añadir etiqueta
            label = Label(x=x_section, y=y_section + 5,
                         text=f"{sensor['name']}\n{sensor['value_attr']}",
                         text_font_size='8pt',
                         text_baseline='bottom',
                         text_align='center')
            p.add_layout(label)
        
        # Guardar output
        output_file(output_path)
        save(p)

if __name__ == '__main__':
    # Generar DXF de prueba
    # create_test_dxf("test_section.dxf")
    
    # Configurar datos de prueba
    test_line_start = (279000, 8665000)  # UTM ficticio
    test_line_end = (279200, 8665000)     # 200 metros de longitud
    base_elev = 400.0  # metros
    
    test_sensors = pd.DataFrame({
        'name': ['Sensor Lima', 'Sensor Cusco', 'Sensor Arequipa'],
        'sensor_type': ['PCV', 'PCT', 'SACV'],
        'east': [279033, 279150, 279180],  # Coordenadas a lo largo de la línea
        'north': [8665000, 8665000, 8665000],  # Mismo norte que la línea
        'zone_number': [18, 18, 18],
        'zone_letter': ['M', 'M', 'M'],
        'name_attr': ['Temperatura', 'Presión', 'Nivel'],
        'value_attr': ['24°C', '1013 hPa', '120 m'],
        'status': ['Activo', 'Inactivo', 'Mantenimiento'],
        'elevation': [200, 250, 300],  # Elevaciones absolutas
        'plot_path': ['plot_lima.html', 'plot_cusco.html', 'plot_arequipa.html']  # Rutas a los gráficos
    })
    
    # Generar gráfico
    plotter = SectionPlotter(test_line_start, test_line_end, base_elev, "test.dxf")
    plotter.plot_section(test_sensors, "seccion_sensores.html")