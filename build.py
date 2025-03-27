import os
import sqlite3
from sqlite3 import Error


def init_directory(directory="./var"):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_connection(db_file):
    """Crear una conexión a la base de datos SQLite"""
    conn = None
    conn = sqlite3.connect(db_file)
    version = sqlite3.sqlite_version
    print(f"Conexión a SQLite {version} exitosa")
    return conn


def read_sql_file(file_path):
    """Leer el contenido de un archivo SQL"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Archivo {file_path} no encontrado")
    except Exception as e:
        print(f"Error leyendo archivo SQL: {e}")
    return None


def execute_query(conn, sql_query, parameters=None):
    """Ejecutar una consulta SQL"""
    try:
        c = conn.cursor()
        # Limpiar y filtrar statements vacíos
        statements = [stmt.strip() for stmt in sql_query.split("--") if stmt.strip()]

        for statement in statements:
            if parameters:
                c.execute(statement, parameters)
            else:
                c.execute(statement)

        # Si es SELECT, devolver resultados y encabezados
        if sql_query.strip().upper().startswith("SELECT"):
            headers = [description[0] for description in c.description]
            return headers, c.fetchall()
        else:
            conn.commit()
            return None, c.rowcount

    except Error as e:
        print(f"Error ejecutando query: {e}")
        return None, None


def init_database(db_path, query_path):

    # Leer consulta SQL desde archivo
    sql_query = read_sql_file(query_path)
    if not sql_query:
        return

    # Conectar a la base de datos
    conn = create_connection(db_path)

    if conn is not None:
        # Ejecutar consulta
        headers, results = execute_query(conn, sql_query)

        # Mostrar resultados
        if headers and results:
            # Imprimir encabezados
            print("\nResultados:")
            print(" | ".join(headers))
            print("-" * (sum(len(h) for h in headers) + 3 * len(headers)))

            # Imprimir datos
            for row in results:
                print(" | ".join(str(item) for item in row))

        elif results is not None:
            print(f"\nQuery ejecutada. Filas afectadas: {results}")

        conn.close()
    else:
        print("Error! No se pudo crear la conexión a la base de datos")


# Create the directories
directories = ["./var", "./data"]

for directory in directories:
    init_directory(directory="./var")

# Create database
db_path = "./data/database/database.db"
query_path = "./data/database/alter.sql"

init_database(db_path=db_path, query_path=query_path)
