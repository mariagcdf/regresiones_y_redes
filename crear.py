# Crear carpetas para cada ejercicio con subcarpetas data y scripts
mkdir -p ejercicio_1_regresion_simple/data ejercicio_1_regresion_simple/scripts
mkdir -p ejercicio_2_regresion_multiple/data ejercicio_2_regresion_multiple/scripts
mkdir -p ejercicio_3_clasificacion_binaria/data ejercicio_3_clasificacion_binaria/scripts
mkdir -p ejercicio_4_redes_neuronales/data ejercicio_4_redes_neuronales/scripts

# Crear archivos base
touch README.md .gitignore requirements.txt

# Crear archivos vacíos de script en cada carpeta (opcional, para que git no ignore las carpetas vacías)
touch ejercicio_1_regresion_simple/scripts/regresion_lineal_simple.py
touch ejercicio_2_regresion_multiple/scripts/regresion_lineal_multiple.py
touch ejercicio_3_clasificacion_binaria/scripts/clasificacion_binaria.py
touch ejercicio_4_redes_neuronales/scripts/redes_neuronales.py
