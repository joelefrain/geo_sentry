import ezdxf

from matplotlib import colormaps
from matplotlib.colors import rgb2hex


def get_unique_marker_convo(df_index, total_dfs, color_palette="viridis"):
    """
    Generate a unique combination of color and marker for a given dataframe index.
    Ensures consistency across series.
    """
    from itertools import cycle

    # Define unique styles for markers
    unique_styles = {"markers": ["o", "s", "D", "v", "^", "<", ">", "p", "h"]}

    # Generate random colors from the colormap
    colormap = colormaps[color_palette]
    if total_dfs == 1:
        color = rgb2hex(
            colormap(0.1)
        )  # Use a fixed value if there's only one dataframe
    else:
        # Calculate equidistant position based on df_index
        pos = df_index / (total_dfs - 1)
        color = rgb2hex(colormap(pos))

    # Cycle through markers to ensure consistency
    marker_cycle = cycle(unique_styles["markers"])
    for _ in range(df_index + 1):
        marker = next(marker_cycle)

    combination = (color, marker)

    return combination


def dxf_color_to_hex(color_number):
    """Convierte el número de color DXF a código hexadecimal."""
    # Mapa de colores DXF básicos
    dxf_colors = {
        1: "#FF0000",  # Red
        2: "#FFFF00",  # Yellow
        3: "#00FF00",  # Green
        4: "#00FFFF",  # Cyan
        5: "#0000FF",  # Blue
        6: "#FF00FF",  # Magenta
        7: "#000000",  # Black
        8: "#808080",  # Gray
        9: "#C0C0C0",  # Light Gray
    }
    return dxf_colors.get(color_number, "#000000")  # Negro por defecto


def dxf_linetype_to_style(linetype: str) -> str:
    """
    Convierte el tipo de línea DXF (string) a un estilo de línea de matplotlib.
    Usa un mapeo basado en los nombres estándar de AutoCAD.

    Ejemplos:
    - "CONTINUOUS" -> "-"
    - "DASHED" -> "--"
    - "DOTTED", "DOT" -> ":"
    - "DASHDOT" -> "-."
    - Cualquier otro no reconocido -> "-"
    """
    if not linetype:
        return "-"

    linetype = linetype.strip().lower()

    dxf_linetypes = {
        "bylayer": "-",
        "byblock": "-",
        "continuous": "-",
        "solid": "-",
        "dashed": "--",
        "hidden": "--",
        "center": "--",
        "phantom": "--",
        "border": "--",
        "dotted": ":",
        "dot": ":",
        "dashdot": "-.",
        "center2": "-.",
        "dashdot2": "-.",
        "chain": "-.",
    }

    # Buscar coincidencia exacta
    if linetype in dxf_linetypes:
        return dxf_linetypes[linetype]

    # Buscar coincidencia parcial (fallback)
    for key in dxf_linetypes:
        if key in linetype:
            return dxf_linetypes[key]

    # Estilo por defecto
    return "-"


class dxfParser:
    def __init__(self, dxf_path):
        self.dxf_path = dxf_path

    def parse_entities(self):
        """
        Extrae todas las entidades relevantes del DXF y devuelve una lista de diccionarios
        con geometría, color, grosor y estilo de línea.
        """
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()
        entities = []

        def get_linewidth(entity):
            return (
                (entity.dxf.lineweight / 100)
                if entity.dxf.hasattr("lineweight") and entity.dxf.lineweight > 0
                else 1
            )

        def get_color(entity):
            return dxf_color_to_hex(entity.dxf.color)

        def get_linestyle(entity):
            return (
                dxf_linetype_to_style(entity.dxf.linetype)
                if entity.dxf.hasattr("linetype")
                else "-"
            )

        # LINE
        for entity in msp.query("LINE"):
            entities.append(
                {
                    "type": "LINE",
                    "points": [
                        (entity.dxf.start[0], entity.dxf.start[1]),
                        (entity.dxf.end[0], entity.dxf.end[1]),
                    ],
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                    "linestyle": get_linestyle(entity),
                }
            )

        # LWPOLYLINE
        for entity in msp.query("LWPOLYLINE"):
            points = [(p[0], p[1]) for p in entity.get_points()]
            entities.append(
                {
                    "type": "LWPOLYLINE",
                    "points": points,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                    "linestyle": get_linestyle(entity),
                    "closed": entity.closed,
                }
            )

        # POLYLINE
        for entity in msp.query("POLYLINE"):
            points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices()]
            entities.append(
                {
                    "type": "POLYLINE",
                    "points": points,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                    "linestyle": get_linestyle(entity),
                    "closed": entity.is_closed,
                }
            )

        # HATCH
        for entity in msp.query("HATCH"):
            color = get_color(entity)
            for path in entity.paths:
                if hasattr(path, "polylines"):
                    for poly in path.polylines:
                        entities.append(
                            {
                                "type": "HATCH",
                                "points": list(poly),
                                "color": color,
                                "alpha": 1.0,
                            }
                        )
                elif hasattr(path, "edges"):
                    for edge in path.edges:
                        if hasattr(edge, "start") and hasattr(edge, "end"):
                            entities.append(
                                {
                                    "type": "HATCH_EDGE",
                                    "points": [edge.start, edge.end],
                                    "color": color,
                                    "linewidth": 1,
                                    "linestyle": "-",
                                }
                            )

        # ARC
        for entity in msp.query("ARC"):
            entities.append(
                {
                    "type": "ARC",
                    "center": (entity.dxf.center[0], entity.dxf.center[1]),
                    "radius": entity.dxf.radius,
                    "start_angle": entity.dxf.start_angle,
                    "end_angle": entity.dxf.end_angle,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                }
            )

        # CIRCLE
        for entity in msp.query("CIRCLE"):
            entities.append(
                {
                    "type": "CIRCLE",
                    "center": (entity.dxf.center[0], entity.dxf.center[1]),
                    "radius": entity.dxf.radius,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                }
            )

        # SPLINE
        for entity in msp.query("SPLINE"):
            fit_points = [(p[0], p[1]) for p in entity.fit_points]
            entities.append(
                {
                    "type": "SPLINE",
                    "points": fit_points,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                    "linestyle": get_linestyle(entity),
                }
            )

        # ELLIPSE
        for entity in msp.query("ELLIPSE"):
            try:
                ellipse_points = [
                    entity.point_at(t)[:2] for t in [i / 100.0 for i in range(101)]
                ]
            except Exception:
                ellipse_points = []
            entities.append(
                {
                    "type": "ELLIPSE",
                    "points": ellipse_points,
                    "color": get_color(entity),
                    "linewidth": get_linewidth(entity),
                    "linestyle": get_linestyle(entity),
                }
            )

        # POINT
        for entity in msp.query("POINT"):
            entities.append(
                {
                    "type": "POINT",
                    "point": (entity.dxf.location.x, entity.dxf.location.y),
                    "color": get_color(entity),
                }
            )

        # TEXT
        for entity in msp.query("TEXT"):
            entities.append(
                {
                    "type": "TEXT",
                    "position": (entity.dxf.insert[0], entity.dxf.insert[1]),
                    "text": entity.dxf.text,
                    "color": get_color(entity),
                    "height": getattr(entity.dxf, "height", 8),
                }
            )

        return entities

    def _parse_dxf(self):
        """Extrae los puntos del perfil del terreno del DXF."""
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()

        all_points = []
        lines_data = []

        # Extraer todas las líneas, polylines, hatch y texto
        for entity in msp.query("LINE LWPOLYLINE HATCH TEXT"):
            if entity.dxftype() == "LINE":
                points = [
                    (entity.dxf.start.x, entity.dxf.start.y),
                    (entity.dxf.end.x, entity.dxf.end.y),
                ]
                color = dxf_color_to_hex(entity.dxf.color)
            if entity.dxftype() == "LWPOLYLINE":
                points = [(vertex[0], vertex[1]) for vertex in entity.vertices()]
                color = dxf_color_to_hex(entity.dxf.color)
            if entity.dxftype() == "HATCH":
                # Para HATCH, no hay puntos directos, pero podemos usar el bounding box
                bbox = entity.bounding_box()
                points = [
                    (bbox.extmin.x, bbox.extmin.y),
                    (bbox.extmax.x, bbox.extmax.y),
                ]
                color = dxf_color_to_hex(entity.dxf.color)
            if entity.dxftype() == "TEXT":
                # Para TEXT, usamos la posición como un punto
                points = [(entity.dxf.insert.x, entity.dxf.insert.y)]
                color = dxf_color_to_hex(entity.dxf.color)

            all_points.extend(points)
            lines_data.append({"points": points, "color": color})

        if not lines_data:
            raise ValueError("No se encontraron entidades válidas en el DXF")

        return lines_data
