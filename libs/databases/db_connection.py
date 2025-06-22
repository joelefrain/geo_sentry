import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import mysql.connector

from dotenv import load_dotenv
from mysql.connector import Error

from libs.utils.config_variables import ENV_FILE_PATH
from libs.utils.config_logger import get_logger

logger = get_logger("libs.databases.db_connection")


class DatabaseConnection:
    def __init__(self):
        self.connection = None

        # Cargar variables de entorno desde el archivo .env
        load_dotenv(ENV_FILE_PATH)
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_NAME")

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                auth_plugin="mysql_native_password",
            )
            if self.connection.is_connected():
                logger.info("Conexión exitosa a la base de datos.")
                return self.connection
            else:
                logger.error("No se pudo conectar a la base de datos.")
                return None
        except Error as ex:
            logger.exception(f"Error al conectar a la base de datos: {ex}")
            return None

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            logger.info("Conexión a la base de datos cerrada.")
