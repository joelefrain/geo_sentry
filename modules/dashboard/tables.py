<<<<<<< HEAD
import pandas as pd
import toml
from pathlib import Path
from abc import ABC
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    TableColumn,
    DateFormatter,
    NumberFormatter,
    StringFormatter,
    Select,
    CustomJS,
)


class BaseTable(ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        height: int = 600,
        instrument: str = None,
    ):
        self.df = df
        self.height = height
        self.instrument = instrument
        self.instrument_config_path = Path(__file__).parent.parent / f"calculations/data/{self.instrument}.toml"
        self.config = toml.load(self.instrument_config_path)
        self.spanish_names = self.config["names"]["es"]
        
        self.source = ColumnDataSource(data=self.df)
        self.table = self._initialize_table()

    def _create_table_columns(self) -> list[TableColumn]:
        columns = []
        for col in self.df.columns:
            if col not in self.spanish_names:
                continue  # Skip columns not in the toml configuration
            
            label = self.spanish_names.get(col, col)
            
            if col == 'time':
                formatter = DateFormatter(format='%Y-%m-%d %H:%M:%S')
                width = 180
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                formatter = NumberFormatter(format='0.000')
                width = 120
            else:
                formatter = StringFormatter()
                width = 150
            
            columns.append(TableColumn(
                field=col,
                title=label,
                formatter=formatter,
                sortable=True,
                width=width,
            ))
        return columns

    def _initialize_table(self):
        return DataTable(
            source=self.source,
            columns=self._create_table_columns(),
            height=self.height,
            sizing_mode="stretch_width",
            index_position=None,
            styles={
                'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
                'font-size': '13px',
                'background-color': '#ffffff',
                'border': '1px solid #e0e0e0',
                'border-radius': '4px',
            },
            row_height=35,
            css_classes=['custom-table'],
            width=None,
            autosize_mode='none',
            background='#ffffff',
            min_height=200,
        )

    @property
    def layout(self):
        style = """
        <style>
            .custom-table .slick-header-column {
                background-color: #343a40 !important;
                border-right: 1px solid #dee2e6 !important;
                border-bottom: 2px solid #dee2e6 !important;
                padding: 8px !important;
                font-weight: 600 !important;
                color: #ffffff !important;
                text-align: center !important;
            }
            .custom-table .slick-cell {
                border-right: 1px solid #dee2e6 !important;
                border-bottom: 1px solid #dee2e6 !important;
                padding: 8px !important;
                color: #212529 !important;
            }
            .custom-table .slick-row:hover {
                background-color: #e9ecef !important;
            }
            .custom-table .slick-row.odd {
                background-color: #f8f9fa !important;
            }
            .custom-table .slick-row.even {
                background-color: #ffffff !important;
            }
            .custom-table .slick-row.selected {
                background-color: #adb5bd !important;
            }
        </style>
        """
        from bokeh.models import Div
        return column(
            Div(text=style),
            self.table,
            sizing_mode="stretch_width",
            spacing=10
        )


class MultiDataFrameTable(BaseTable):
    def __init__(self, dfs: dict[str, pd.DataFrame], **kwargs):
        self.df_dict = dfs
        first_df = next(iter(dfs.values()))
        super().__init__(first_df, **kwargs)
        self.df_select = self._create_df_selector()

    def _create_df_selector(self):
        df_select = Select(
            title="Seleccionar serie temporal:",
            value=list(self.df_dict.keys())[0],
            options=list(self.df_dict.keys()),
        )
        
        callback = CustomJS(
            args={
                'source': self.source,
                'df_select': df_select,
                'data_dict': {name: ColumnDataSource(data=df).data for name, df in self.df_dict.items()}
            },
            code="""
            const new_data = data_dict[df_select.value];
            source.data = {...new_data};
            """
        )
        
        df_select.js_on_change('value', callback)
        return df_select

    @property
    def layout(self):
        return column(self.df_select, self.table, sizing_mode="stretch_width")


def save_table(layout: column, output_path: str) -> None:
    from bokeh.io import save, output_file
    output_file(output_path, title=Path(output_path).stem)
    save(layout)


if __name__ == "__main__":
    # Example usage
    csv_path = Path(r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\process\ShahuinoSAC\Shahuindo\gold\PCT\PCT.DME_CHO_DIQ.D-CH-1.csv")
    csv_path_2 = Path(r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\process\ShahuinoSAC\Shahuindo\gold\PCT\PCT.DME_CHO_DIQ.D-CH-2.csv")

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    df_2 = pd.read_csv(csv_path_2)
    df_2['time'] = pd.to_datetime(df_2['time'])

    # Single table example
    table = BaseTable(df, instrument="prism")
    save_table(table.layout, "table.html")

    # Multi DataFrame table example
    table = MultiDataFrameTable(
        dfs={"Table 1": df, "Table 2": df_2},
        instrument="prism",
    )
    save_table(table.layout, "multi_table.html")
=======
import pandas as pd
import toml
from pathlib import Path
from abc import ABC
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    TableColumn,
    DateFormatter,
    NumberFormatter,
    StringFormatter,
    Select,
    CustomJS,
)


class BaseTable(ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        height: int = 600,
        instrument: str = None,
    ):
        self.df = df
        self.height = height
        self.instrument = instrument
        self.instrument_config_path = Path(__file__).parent.parent / f"calculations/data/{self.instrument}.toml"
        self.config = toml.load(self.instrument_config_path)
        self.spanish_names = self.config["names"]["es"]
        
        self.source = ColumnDataSource(data=self.df)
        self.table = self._initialize_table()

    def _create_table_columns(self) -> list[TableColumn]:
        columns = []
        for col in self.df.columns:
            if col not in self.spanish_names:
                continue  # Skip columns not in the toml configuration
            
            label = self.spanish_names.get(col, col)
            
            if col == 'time':
                formatter = DateFormatter(format='%Y-%m-%d %H:%M:%S')
                width = 180
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                formatter = NumberFormatter(format='0.000')
                width = 120
            else:
                formatter = StringFormatter()
                width = 150
            
            columns.append(TableColumn(
                field=col,
                title=label,
                formatter=formatter,
                sortable=True,
                width=width,
            ))
        return columns

    def _initialize_table(self):
        return DataTable(
            source=self.source,
            columns=self._create_table_columns(),
            height=self.height,
            sizing_mode="stretch_width",
            index_position=None,
            styles={
                'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
                'font-size': '13px',
                'background-color': '#ffffff',
                'border': '1px solid #e0e0e0',
                'border-radius': '4px',
            },
            row_height=35,
            css_classes=['custom-table'],
            width=None,
            autosize_mode='none',
            background='#ffffff',
            min_height=200,
        )

    @property
    def layout(self):
        style = """
        <style>
            .custom-table .slick-header-column {
                background-color: #343a40 !important;
                border-right: 1px solid #dee2e6 !important;
                border-bottom: 2px solid #dee2e6 !important;
                padding: 8px !important;
                font-weight: 600 !important;
                color: #ffffff !important;
                text-align: center !important;
            }
            .custom-table .slick-cell {
                border-right: 1px solid #dee2e6 !important;
                border-bottom: 1px solid #dee2e6 !important;
                padding: 8px !important;
                color: #212529 !important;
            }
            .custom-table .slick-row:hover {
                background-color: #e9ecef !important;
            }
            .custom-table .slick-row.odd {
                background-color: #f8f9fa !important;
            }
            .custom-table .slick-row.even {
                background-color: #ffffff !important;
            }
            .custom-table .slick-row.selected {
                background-color: #adb5bd !important;
            }
        </style>
        """
        from bokeh.models import Div
        return column(
            Div(text=style),
            self.table,
            sizing_mode="stretch_width",
            spacing=10
        )


class MultiDataFrameTable(BaseTable):
    def __init__(self, dfs: dict[str, pd.DataFrame], **kwargs):
        self.df_dict = dfs
        first_df = next(iter(dfs.values()))
        super().__init__(first_df, **kwargs)
        self.df_select = self._create_df_selector()

    def _create_df_selector(self):
        df_select = Select(
            title="Seleccionar serie temporal:",
            value=list(self.df_dict.keys())[0],
            options=list(self.df_dict.keys()),
        )
        
        callback = CustomJS(
            args={
                'source': self.source,
                'df_select': df_select,
                'data_dict': {name: ColumnDataSource(data=df).data for name, df in self.df_dict.items()}
            },
            code="""
            const new_data = data_dict[df_select.value];
            source.data = {...new_data};
            """
        )
        
        df_select.js_on_change('value', callback)
        return df_select

    @property
    def layout(self):
        return column(self.df_select, self.table, sizing_mode="stretch_width")


def save_table(layout: column, output_path: str) -> None:
    from bokeh.io import save, output_file
    output_file(output_path, title=Path(output_path).stem)
    save(layout)


if __name__ == "__main__":
    # Example usage
    csv_path = Path(r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\process\ShahuinoSAC\Shahuindo\gold\PCT\PCT.DME_CHO_DIQ.D-CH-1.csv")
    csv_path_2 = Path(r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\process\ShahuinoSAC\Shahuindo\gold\PCT\PCT.DME_CHO_DIQ.D-CH-2.csv")

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    df_2 = pd.read_csv(csv_path_2)
    df_2['time'] = pd.to_datetime(df_2['time'])

    # Single table example
    table = BaseTable(df, instrument="prism")
    save_table(table.layout, "table.html")

    # Multi DataFrame table example
    table = MultiDataFrameTable(
        dfs={"Table 1": df, "Table 2": df_2},
        instrument="prism",
    )
    save_table(table.layout, "multi_table.html")
>>>>>>> 118aabc (update | Independizacion del locale del sistema operativo)
