import matplotlib.pyplot as plt
import locale


class PlotConfig:
    """Class to configure global matplotlib parameters."""

    @classmethod
    def setup_matplotlib(cls):
        """Configure global matplotlib parameters.

        Parameters
        ----------
        use_date_format : bool, optional
            Whether to use date format for x-axis, by default False.
        x_data_range : tuple, optional
            The range of x-axis data (start, end), by default None.
        """

        plt.rcParams["backend"] = "Agg"
        
        # Set font family
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 8

        # Enable locale settings for number formatting
        plt.rcParams["axes.formatter.use_locale"] = True
        locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")

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
        plt.rcParams["axes.xmargin"] = 0
        plt.rcParams["axes.ymargin"] = 0.20
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
        plt.rcParams["date.autoformatter.day"] = "%d-%m-%y"
        plt.rcParams["date.autoformatter.month"] = "%d-%m-%y"
        plt.rcParams["date.autoformatter.year"] = "%d-%m-%y"

        # Configure ticks
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.minor.visible"] = False
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
