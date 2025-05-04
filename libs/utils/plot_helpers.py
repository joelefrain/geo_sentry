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