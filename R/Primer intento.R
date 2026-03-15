# =============================================
# CARGAR LIBRERÍAS
# =============================================
library(tidyverse)
library(readr)
library(dplyr)

# Cargar librería PRIMERO (obligatorio)
library(tidyverse)

# Verificar archivos disponibles
list.files("data/")

# Cargar CSVs (ajusta los nombres según lo que veas arriba)
por_anio        <- read_csv("data/03_por_anio.csv")
por_pais        <- read_csv("data/04_por_pais_continente.csv")
por_revista     <- read_csv("data/05_por_revista.csv")
tipo_articulo   <- read_csv("data/06_revision_vs_investigacion_14069.csv")
antibioticos    <- read_csv("data/07_antibioticos_adsorbentes_14069.csv")
fisico_quim     <- read_csv("data/08_variables_fisicoquimicas_14069.csv")
cuantitativos   <- read_csv("data/09_datos_cuantitativos_4674.csv")
bibliometrico   <- read_csv("data/10_bibliometrico_14069.csv")

# Verificar que cargaron bien
names(por_anio)
names(por_pais)
names(bibliometrico)
