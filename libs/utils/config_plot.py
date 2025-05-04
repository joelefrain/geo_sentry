from .config_variables import DECIMAL_CHAR, THOUSAND_CHAR, DEFAULT_FONT, DATE_FORMAT

import matplotlib.pyplot as plt


class PlotConfig:
    """Class to configure global matplotlib parameters."""

    @classmethod
    def setup_matplotlib(cls, ts_serie=True, ymargin=0.20):

        """Configure global matplotlib parameters."""
        # Set system locale
        plt.rcParams["axes.formatter.use_locale"] = True

        # Disable interactive mode and set backend to Agg for non-interactive plotting
        plt.rcParams["backend"] = "Agg"

        # Set font family
        plt.rcParams["font.family"] = DEFAULT_FONT
        plt.rcParams["font.size"] = 8

        # Configure figure appearance
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams["figure.constrained_layout.h_pad"] = 0
        plt.rcParams["figure.constrained_layout.hspace"] = 0
        plt.rcParams["figure.constrained_layout.w_pad"] = 0
        plt.rcParams["figure.constrained_layout.wspace"] = 0
        plt.rcParams["figure.edgecolor"] = "None"
        plt.rcParams["figure.facecolor"] = "None"
        plt.rcParams["figure.titlesize"] = 10
        plt.rcParams["figure.titleweight"] = "bold"
        plt.rcParams["figure.autolayout"] = True

        # Configure axes appearance
        plt.rcParams["axes.facecolor"] = "None"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.ymargin"] = ymargin
        plt.rcParams["axes.spines.bottom"] = True
        plt.rcParams["axes.spines.left"] = True
        plt.rcParams["axes.spines.right"] = True
        plt.rcParams["axes.spines.top"] = True
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams["axes.labelsize"] = 9
        plt.rcParams["axes.titlepad"] = 4
        plt.rcParams["axes.titlelocation"] = "center"

        # Configure date formatting
        plt.rcParams["date.autoformatter.day"] = DATE_FORMAT
        plt.rcParams["date.autoformatter.month"] = DATE_FORMAT
        plt.rcParams["date.autoformatter.year"] = DATE_FORMAT

        # Configure ticks
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["xtick.labelsize"] = 8

        # Configure legend
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["legend.facecolor"] = "white"
        plt.rcParams["legend.framealpha"] = 0.15
        plt.rcParams["legend.edgecolor"] = "None"
        plt.rcParams["legend.fancybox"] = False

        # Configure grid
        plt.rcParams["grid.alpha"] = 0.25
        plt.rcParams["grid.color"] = "gray"
        plt.rcParams["grid.linestyle"] = "-"
        plt.rcParams["grid.linewidth"] = 0.05

        if ts_serie:
            plt.rcParams["axes.xmargin"] = 0
            plt.rcParams["xtick.minor.visible"] = False
        else:
            plt.rcParams["axes.xmargin"] = 0.05
            plt.rcParams["xtick.minor.visible"] = True
