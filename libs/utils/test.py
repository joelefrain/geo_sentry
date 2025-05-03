import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class PlotConfig:
    """Clase para configurar parámetros globales de matplotlib y aplicar locales"""

    def __init__(self, ts_serie=True, ymargin=0.20, lang="fr"):
        """Inicializar la configuración de matplotlib y aplicar formato local automáticamente."""
        # Configurar matplotlib
        self.setup_matplotlib(ts_serie=ts_serie, ymargin=ymargin, lang=lang)

    @classmethod
    def setup_matplotlib(cls, ts_serie=True, ymargin=0.20, lang="fr"):
        """Configurar los parámetros globales de matplotlib y el formato del locale."""
        
        # Establecer formato temporalmente según el idioma deseado
        cls.locale_format = cls.get_locale_format(lang)
        
        # Configuraciones de matplotlib
        plt.rcParams["axes.formatter.use_locale"] = False  # Desactivar formato automático de locale
        # plt.rcParams["backend"] = "Agg"
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 8
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.ymargin"] = ymargin
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = 8

    @classmethod
    def get_locale_format(cls, lang):
        """Simular formato de números según el idioma, devolviendo una función para formatear."""
        if lang == "fr":
            # Comas como separadores decimales y espacio para miles
            return lambda x: '{:,.2f}'.format(x).replace(',', ' ').replace('.', ',')
        elif lang == "en":
            # Comas como separadores decimales y punto para miles
            return lambda x: '{:,.2f}'.format(x)
        else:
            # Formato por defecto sin separadores
            return lambda x: '{:.2f}'.format(x)

    @staticmethod
    def wrapper_formatting(func):
        """Wrapper para aplicar el formateo de números a los ejes del gráfico."""

        def wrapped(ax, *args, **kwargs):
            # Aplicar el formateo del locale antes de la función original
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: PlotConfig.locale_format(x)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: PlotConfig.locale_format(x)))
            return func(ax, *args, **kwargs)
        
        return wrapped

# Crear una instancia de PlotConfig, que automáticamente aplica el formato
plot_config = PlotConfig(lang="fr")

# Crear un gráfico para ver el efecto del formato
fig, ax = plt.subplots()
ax.plot([1000, 2000, 3000, 4000], [1.2345, 2.3456, 3.4567, 4.5678])

# Usar el wrapper para aplicar automáticamente el formateo de números
PlotConfig.wrapper_formatting(lambda ax: None)(ax)  # Llamada de ejemplo (vacío porque solo aplica formateo)

plt.show()
