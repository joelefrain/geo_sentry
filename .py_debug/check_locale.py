import os
import locale

# Simula un locale no existente
os.environ['LC_ALL'] = 'fr_FR.UTF-8'
try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except locale.Error:
    print("Locale no válido o no generado en el sistema, ignorando...")
