# Librerías
import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------------
# Clase para el notebook '1) Generador_Datos.ipynb'

class Generador_Datos:
    # Constructor
    def __init__(self, path_inputs: str, path_outputs: str) -> None:
        """
        Constructor de la clase
        :param path_inputs: Path donde se encuentran los archivos de entrada
        :param path_outputs: Path donde se guardarán los archivos de salida
        """
        # Diccionario para hacer los cambios de los meses
        self.diccionario_mes = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                                }
        # Path para leer el archivo
        self.path_inputs = path_inputs
        # Path para guardar los archivos
        self.path_outputs = path_outputs
    
    # Método para generar el archivo 'unemployment_data_usa'
    def gen_unemployment_data(self, exportar: bool = True) -> pd.DataFrame:
        """
        Método para generar el archivo 'unemployment_data_usa'
        :param exportar: Booleano para exportar el archivo
        :return: Df con los datos de desempleo
        """
        # Cargando CSV
        unemployemnt_data_usa = pd.read_csv(fr'{self.path_inputs}/Unemployment_Data_us.csv')
        # Día genérico
        unemployemnt_data_usa['Day'] = 1
        # Obteniendo el número de mes en vez del str
        unemployemnt_data_usa['Month'] = unemployemnt_data_usa['Month'].apply(lambda x: self.diccionario_mes.get(x))
        # Ordenando el dataset y reiniciando el índice
        unemployemnt_data_usa = unemployemnt_data_usa.sort_values(['Year', 'Month']).reset_index(drop=True)
        # Obteniendo la fecha
        unemployemnt_data_usa[['Year', 'Month', 'Day']] = unemployemnt_data_usa[['Year', 'Month', 'Day']].astype(str)
        unemployemnt_data_usa['Date'] = unemployemnt_data_usa[['Year', 'Month', 'Day']].agg('-'.join, axis=1)
        unemployemnt_data_usa['Date'] = unemployemnt_data_usa['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        # Seleccionando las columnas que necesitamos
        unemployemnt_data_usa = unemployemnt_data_usa[['Date', 'Primary_School', 'High_School', 'Associates_Degree', 'Professional_Degree',
                                                    'White', 'Black', 'Asian', 'Hispanic', 'Men', 'Women']]
        # Cambiando los nombres de las columnas
        unemployemnt_data_usa.columns = ['Fecha', 'Escuela_Primaria', 'Preparatoria', 'Tecnico', 'Profesional',
                                        'Blanco', 'Negro', 'Asiatico', 'Hispano', 'Hombre', 'Mujer']
        # Quitando el año 2020 porque tienen NaN
        unemployemnt_data_usa = unemployemnt_data_usa[unemployemnt_data_usa['Fecha'] < datetime(2020, 1, 1)]
        # Condicional para exportar
        if exportar:
            unemployemnt_data_usa.to_excel(fr'{self.path_outputs}/Desempleo_Por_Grupo.xlsx', index=False)
        return unemployemnt_data_usa
    
    # Método para generar 'Población por Raza'
    @staticmethod
    def get_population_by_race(path_files: str) -> pd.DataFrame:
        """
        Método para obtener la población por raza
        :param path_files: Path a la carpeta donde están los archivos
        :return df_total: Df en donde se encuentran los datos por raza de todo el país
        :return df_estados: Df en donde se encuentran los datos por raza por estado
        """
        # Df que contendrá los datos de los estados
        df = pd.DataFrame()
        # Ciclo para crear un archivo parquet con la concatenación de los csv
        for file in os.listdir(path_files):
            # Solo los archivos .csv
            if file.endswith('.csv'):
                # Obteniendo el año del archivo
                anio = '20' + file.split('.')[0].split('_')[-1]
                # Cargando el archivo
                temp = pd.read_csv(path_files + file)
                # Agregando el año a una columna
                temp['Year'] = anio
                # Concatenando todos los archivos
                df = pd.concat([df, temp], axis=0)
        # Eliminando columnas que no usaremos
        df.drop(['American Indian/Alaska Native', 'Native Hawaiian/Other Pacific Islander', 'Multiple Races', 'Footnotes'], axis=1, inplace=True)
        # Ordenando las columnas
        df = df[['Location', 'Year', 'White', 'Black', 'Asian', 'Hispanic']]
        # Renombrando las columnas
        df.columns = ['Estado', 'Year', 'Blanco_Cantidad', 'Negro_Cantidad', 'Asiatico_Cantidad', 'Hispano_Cantidad']
        # Quitando 'United States' porque no es un estado y Puerto Rico
        df_estados = df[~df['Estado'].isin(['United States', 'Puerto Rico'])].reset_index(drop=True)
        # Tomando solo 'United States' porque es el total de la población
        df_total = df[df['Estado'] == 'United States'].reset_index(drop=True)
        # Regresando el Df
        return df_total, df_estados
    
    # Método para generar los datos por género
    @staticmethod
    def get_population_by_gender(path_files: str):
        """
        Método para obtener los datos de la población por género
        :param path_files: Path a la carpeta donde están los archivos
        :return df_total_gender: Df en donde se encuentran los datos por gérnero de todo el país
        :return df_estados_gender: Df en donde se encuentran los datos por raza por estado
        """
        # Df que contendrá los datos de los estados
        df_gender = pd.DataFrame()
        # Ciclo para crear un archivo parquet con la concatenación de los csv
        for file in os.listdir(path_files):
            # Solo los archivos .csv
            if file.endswith('.csv'):
                # Obteniendo el año del archivo
                anio = '20' + file.split('.')[0].split('_')[-1]
                # Cargando el archivo
                temp = pd.read_csv(path_files + file)
                # Agregando el año a una columna
                temp['Year'] = anio
                # Concatenando todos los archivos
                df_gender = pd.concat([df_gender, temp], axis=0)
        # Eliminando columnas que no usaremos
        #df.drop(['Footnotes'], axis=1, inplace=True)
        # Ordenando las columnas
        df_gender = df_gender[['Location', 'Year', 'Male', 'Female']]
        # Renombrando las columnas
        df_gender.columns = ['Estado', 'Year', 'Hombre_Cantidad', 'Mujer_Cantidad']
        # Quitando 'United States' porque no es un estado y Puerto Rico
        df_estados_gender = df_gender[~df_gender['Estado'].isin(['United States', 'Puerto Rico'])].reset_index(drop=True)
        # Tomando solo 'United States' porque es el total de la población
        df_total_gender = df_gender[df_gender['Estado'] == 'United States'].reset_index(drop=True)
        # Regresando el Df
        return df_total_gender, df_estados_gender
    
    # Método para juntar los dfs de población por raza y género
    def merge_dfs(self, df_total: pd.DataFrame, df_total_gender: pd.DataFrame, df_estados: pd.DataFrame, df_estados_gender: pd.DataFrame, exportar: bool = True) -> pd.DataFrame:
        """
        Método para juntar los dfs de población por raza y género
        :param df_total: Df en donde se encuentran los datos por raza de todo el país
        :param df_total_gender: Df en donde se encuentran los datos por género de todo el país
        :param df_estados: Df en donde se encuentran los datos por raza por estado
        :param df_estados_gender: Df en donde se encuentran los datos por género por estado
        :param exportar: Booleano para exportar los archivos
        :return df_total_gen: Df en donde se encuentran los datos por raza y género de todo el país
        :return df_estados_gen: Df en donde se encuentran los datos por raza y género por estado
        """
        # Junta los dfs de población por raza y género del país
        df_total_gen = df_total.merge(df_total_gender, on=['Estado', 'Year'], how='inner')
        # Junta los dfs de población por raza y género de los estados
        df_estados_gen = df_estados.merge(df_estados_gender, on=['Estado', 'Year'], how='inner')
        # Condicional para exportar
        if exportar:
            df_total_gen.to_excel(fr'{self.path_outputs}/Poblacion_Total.xlsx', index=False)
            df_estados_gen.to_excel(fr'{self.path_outputs}/Poblacion_Estados.xlsx', index=False)
        # Regresa los dfs
        return df_total_gen, df_estados_gen
    
    # Método para obtener los datos de de desempleo
    def get_unemployment(self) -> pd.DataFrame:
        """
        Método para obtener los datos de de desempleo
        :return df: Df en donde se encuentran los datos de desempleo
        """
        # Cargando Dataset General
        desempleo = pd.read_excel(fr'{self.path_outputs}/Desempleo_Por_Grupo.xlsx')
        # Sacando porcentaje para las columnas[ Blanco, Negro, Asiatico, Hispano, Hombre, Mujer] con apply
        desempleo[['Blanco', 'Negro', 'Asiatico', 'Hispano', 'Hombre', 'Mujer']] = desempleo[['Blanco', 'Negro', 'Asiatico', 'Hispano', 'Hombre', 'Mujer']].apply(lambda x: x / 100)
        # Sacando el año de la fecha
        desempleo['Year'] = desempleo['Fecha'].apply(lambda x: x.year)
        # Seleccionando columnas
        desempleo = desempleo[['Fecha', 'Year', 'Blanco', 'Negro', 'Asiatico', 'Hispano', 'Hombre', 'Mujer']]
        # Regresando el Df
        return desempleo
    
    # Método para obtener el DF general
    def get_general_data(self, desempleo, df, exportar: bool = True) -> pd.DataFrame:
        """
        Método para obtener el DF final
        :param desempleo: DF con los datos sobre desempleo
        :param df: Df con los datos generales
        :param exportar: Booleano para exportar el dataset
        :return df_general: Df con los datos finales
        """
        # Poniendo los años en int
        desempleo['Year'] = desempleo['Year'].astype(int)
        df['Year'] = df['Year'].astype(int)
        # DF general
        df_general = desempleo.merge(df, on=['Year'], how='inner')
        # Multiplicaciones porcentual por valor
        df_general['Blanco_Cantidad'] = (df_general['Blanco'] * df_general['Blanco_Cantidad']).astype(int)
        df_general['Negro_Cantidad'] = (df_general['Negro'] * df_general['Negro_Cantidad']).astype(int)
        df_general['Asiatico_Cantidad'] = (df_general['Asiatico'] * df_general['Asiatico_Cantidad']).astype(int)
        df_general['Hispano_Cantidad'] = (df_general['Hispano'] * df_general['Hispano_Cantidad']).astype(int)
        df_general['Hombre_Cantidad'] = (df_general['Hombre'] * df_general['Hombre_Cantidad']).astype(int)
        df_general['Mujer_Cantidad'] = (df_general['Mujer'] * df_general['Mujer_Cantidad']).astype(int)
        # Multiplicando por 100 para obtener el porcentaje
        df_general['Blanco'] = df_general['Blanco'] * 100
        df_general['Negro'] = df_general['Negro'] * 100
        df_general['Asiatico'] = df_general['Asiatico'] * 100
        df_general['Hispano'] = df_general['Hispano'] * 100
        df_general['Hombre'] = df_general['Hombre'] * 100
        df_general['Mujer'] = df_general['Mujer'] * 100
        # Condicional
        if exportar:
            df_general.to_excel(fr'{self.path_outputs}/Desempleo_USA_General.xlsx', index=False)
        # Regresando el Df
        return df_general

# -----------------------------------------------------------------------------------------------------------------------------------
# Funciones y Clase para el EDA

# Función para obtener las etiquetas automáticamente
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,
                1.05*height,
                '%d'%int(height),
                ha='center', va='bottom')

# Función para obtener el color a usar
def _get_colors_to_use(variables):
    colors = plt.cm.jet(np.linspace(0, 1, len(variables)))
    return dict(zip(variables, colors))

# Clase para el análisis de variables
class VariableAnalysis:
    # Constructor
    def __init__(self, df, pattern, exception_patterns=[], extra_cols=None):
        """
        Constructor de la clase
        :param df: Df con los datos
        :param pattern: Patrón para obtener las columnas
        :param exception_patterns: Patrones para excluir columnas
        :param extra_cols: Columnas extra a incluir
        """
        self.df = df
        self.pattern = pattern
        self.extra_cols = extra_cols
        self.exception_patterns = exception_patterns
        self.data_transformed = False
    
    @staticmethod
    def _clean_columns(df):
        """
        Método para limpiar las columnas
        :param df: Df con los datos
        :return df: Df con las columnas limpias
        """
        df.columns = [str(s).strip().replace(' ', '_')
            for s in df.columns]
        return df
    
    @staticmethod
    def _replace_valus_on_specific_columns(df, patterns,
                                        exception_patterns,
                                        extra_cols_with_no_pattern):
        """
        Método para reemplazar valores en columnas específicas
        :param df: Df con los datos
        :param patterns: Patrones para obtener las columnas
        :param exception_patterns: Patrones para excluir columnas
        :param extra_cols_with_no_pattern: Columnas extra a incluir
        :return df: Df con los valores reemplazados"""
        if extra_cols_with_no_pattern is None:
            extra_cols_with_no_pattern = []
        pattern_cols = []
        for pattern in patterns:
            found_columns = [c for c in df.columns if pattern in c]
            for found_column in found_columns:
                for exception_pattern in exception_patterns:
                    if exception_pattern in found_column:
                        found_columns.remove(found_column)                
            pattern_cols += found_columns        
        pattern_cols += extra_cols_with_no_pattern
        object_cols = df.select_dtypes(include=['object']).columns
        for pattern_col in pattern_cols:
            if pattern_col in object_cols:
                df[pattern_col] = df[pattern_col].apply(
                    lambda x: x.strip().replace(
                        '$','').replace(
                        ',', '').replace('%', '')
                    if isinstance(x, str)
                    else x)
                df[pattern_col].replace({'': None,
                                        '  -   ': None,
                                        '-': None,
                                        'N.A.': None,
                                        'S/D': None}, inplace=True)
                df[pattern_col] = df[pattern_col].astype('float64')
        return df
    
    @staticmethod
    def _basic_stats_for_numerical_variables(df):
        """
        Método para obtener estadísticas básicas de las variables numéricas
        :param df: Df con los datos
        :return stats: Diccionario con las estadísticas
        """
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        stats = {}
        for numeric_column in numeric_df.columns:
            mean = numeric_df[numeric_column].mean()
            median = numeric_df[numeric_column].median()
            std = numeric_df[numeric_column].std()
            quantile25, quantile75 = numeric_df[numeric_column].quantile(
                q=[0.25, 0.75])
            null_count = 100 * (numeric_df[numeric_column].isnull().sum() / len(numeric_df))
            stats[numeric_column] = {'mean': mean,
                                    'median': median,
                                    'std': std,
                                    'q25': quantile25,
                                    'q75': quantile75,
                                    'nulls': null_count
                                    }
        return stats
    
    @staticmethod
    def _basic_stats_for_object_variables(df):
        """
        Método para obtener estadísticas básicas de las variables categóricas
        :param df: Df con los datos
        :return stats: Diccionario con las estadísticas
        """
        object_df = df.select_dtypes(include=['object'])
        stats = {}
        for object_column in object_df.columns:
            # Unique values
            unique_vals = len(object_df[object_column].unique())
            all_values = object_df[object_column].value_counts()
            mode = (all_values.index[0],
                    100 * (all_values.values[0] / len(object_df)))
            null_count = (object_df[object_column].isnull().sum() / len(object_df)) * 100
            stats[object_column] = {'unique_vals': unique_vals,
                                'mode': mode,
                                'null_count': null_count}  
        return stats
    
    def _fit(self):
        """
        Método para preparar los datos
        :return self: Instancia de la clase
        """
        self.df = self._replace_valus_on_specific_columns(
            df=self.df, 
            patterns=self.pattern,
            exception_patterns=self.exception_patterns,
            extra_cols_with_no_pattern=self.extra_cols)    
        self.df = self._clean_columns(df=self.df)
        return self
    
    def _transform(self):
        """
        Método para transformar los datos
        :return numeric_stats: Estadísticas básicas de las variables numéricas
        :return object_stats: Estadísticas básicas de las variables categóricas
        :return df: Df con los datos
        """
        numeric_stats = self._basic_stats_for_numerical_variables(
            df=self.df)
        object_stats = self._basic_stats_for_object_variables(
            df=self.df)
        self.data_transformed = True
        return numeric_stats, object_stats, self.df
    
    def fit_transform(self):
        """
        Método para preparar y transformar los datos
        :return numeric_stats: Estadísticas básicas de las variables numéricas
        :return object_stats: Estadísticas básicas de las variables categóricas
        :return df: Df con los datos
        """
        return self._fit()._transform()

    def plot_numeric(self, df, numeric_stats):
        """
        Método para graficar las variables numéricas
        :param df: Df con los datos
        :param numeric_stats: Estadísticas básicas de las variables numéricas
        :return: None
        """
        if not self.data_transformed:
            raise ValueError('Data has not been prepared. \
            Execute method fit_transform in order to so.')
        corr = df.select_dtypes(exclude=['object']).corr()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.matshow(corr, cmap='Blues')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.grid(False)
        metrics = ['mean', 'median','std', 'q25', 'q75','nulls']
        colors = _get_colors_to_use(metrics)
        for index, variable in enumerate(sorted(numeric_stats.keys())):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            bar_position = -1
            for metric, value in numeric_stats[variable].items():
                bar_position += 1
                if value is None or np.isnan(value):
                    value = -1
                bar_plot = ax[0].bar(bar_position, value, 
                                    label=metric, color=colors[metric])
                autolabel(bar_plot, ax[0])
                df[variable].plot(kind='hist', color='blue',
                                        alpha=0.4, ax=ax[1])
                df.boxplot(ax=ax[2], column=variable)
                ax[0].set_xticks(range(len(metrics)))
                ax[0].set_xticklabels(metrics, rotation=90)
                ax[2].set_xticklabels([], rotation=90)
                ax[0].set_title('\n Basic metrics \n', fontsize=10)
                ax[1].set_title('\n Data histogram \n', fontsize=10)
                ax[2].set_title('\n Data boxplot \n', fontsize=10)
                fig.suptitle(f'Variable: {variable} \n\n\n', fontsize=15)
                fig.tight_layout()
    
    def plot_categorical(self, df, object_stats):
        """
        Método para graficar las variables categóricas
        :param df: Df con los datos
        :param object_stats: Estadísticas básicas de las variables categóricas
        :return: None
        """
        if not self.data_transformed:
            raise ValueError('Data has not been prepared. \
            Execute method fit_transform in order to so.')
        metrics = ['unique_vals', 'mode', 'null_count']
        colors = _get_colors_to_use(metrics)
        for index, variable in enumerate(sorted(object_stats.keys())):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            bar_position = -1
            for metric, value in object_stats[variable].items():
                bar_position += 1
                if metric == 'mode':
                    mode = value[0]
                    value = value[1]
                if value is None or np.isnan(value):
                    value = -1
                bar_plot = ax.bar(bar_position, value, 
                                label=metric, color=colors[metric])
                autolabel(bar_plot, ax)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, rotation=90, fontsize=15)
            ax.set_title(f'\n Basic object metrics: {variable} \n Mode: {mode}\n',
                        fontsize=15)
            fig.tight_layout()

# Función para obtener el skeewness
def skewness(unemployemnt_data_usa):
    #Sesgo
    skewness = round(unemployemnt_data_usa.skew(),2)
    skewness = skewness.to_frame()
    skewness = skewness.rename(columns={0: "value"}) 
    # Función para obtener el skeewness
    def f(x):
        if x['value'] < -1 or x['value'] > 1: return 'Highly Skewed'
        elif (x['value']<=0 and x['value']>=-0.5) or (x['value'] >=0 and x['value']<=0.5):
            return 'Symmetric distribution'
        else: return 'Moderately skewed'
    # Aplicamos la función
    skewness['skewness'] = skewness.apply(f, axis=1)
    return skewness
