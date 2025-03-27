import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import locale

class PlotConfig:
    """Class to configure global matplotlib parameters."""

    # Global matplotlib variables
    font_family = "Arial"
    legend_loc = "upper left"

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

        # Set font family and legend location
        plt.rcParams["font.family"] = cls.font_family
        plt.rcParams["legend.loc"] = cls.legend_loc

        # Enable locale settings for number formatting
        plt.rcParams["axes.formatter.use_locale"] = True

        # Set locale to use comma as decimal separator
        locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")

        # Automatically configure tight_layout for all figures
        plt.rcParams["figure.constrained_layout.h_pad"] = 0
        plt.rcParams["figure.constrained_layout.hspace"] = 0
        plt.rcParams["figure.constrained_layout.w_pad"] = 0
        plt.rcParams["figure.constrained_layout.wspace"] = 0
        plt.rcParams["figure.constrained_layout.use"] = True

        # Set matplotlib backend
        plt.rcParams["backend"] = 'Agg'

        # Set figure edge and face colors
        plt.rcParams["figure.edgecolor"] = 'None'
        plt.rcParams["figure.facecolor"] = 'None'
        
        # Set axes
        plt.rcParams["axes.facecolor"] = 'None'
        plt.rcParams["axes.edgecolor"] = 'black'
        plt.rcParams["axes.grid"] = 'True'
        
        plt.rcParams["date.autoformatter.day"] = '%d-%m-%y'
        plt.rcParams["date.autoformatter.month"] = '%d-%m-%y'
        plt.rcParams["date.autoformatter.year"] = '%d-%m-%y'

        plt.rcParams["axes.xmargin"] = 0
        plt.rcParams["axes.ymargin"] = 0
        # plt.rcParams["axes.autolimit_mode"] = "data"
        
        
        plt.rcParams["axes.spines.bottom"] = "True"
        plt.rcParams["axes.spines.left"] = "True"
        plt.rcParams["axes.spines.right"] = "True"
        plt.rcParams["axes.spines.top"] = "True"

        plt.rcParams["ytick.minor.visible"] = "True"
        plt.rcParams["xtick.minor.visible"] = "True"
        
        plt.rcParams["axes.titleweight"] = "bold"
        
        # Set font sizes
        plt.rcParams["font.size"] = 8
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams["axes.labelsize"] = 9
        plt.rcParams["axes.titlepad"] = 4
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["legend.fontsize"] = 8
        
        # Configure automatic text adjustment for long titles
        plt.rcParams["figure.titlesize"] = 10
        plt.rcParams["axes.titlelocation"] = "center"
        plt.rcParams["figure.titleweight"] = "bold"
        plt.rcParams["figure.autolayout"] = True
        
        plt.rcParams["legend.facecolor"] = "white"
        plt.rcParams["legend.edgecolor"] = "None"
        plt.rcParams["legend.fancybox"] = "False"