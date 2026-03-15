# Revisión Sistemática con Metaanálisis
## Bioadsorbentes frente a antibióticos y hormonas en medios acuosos

## Descripción
Script de screening bibliográfico automatizado para revisión sistemática
siguiendo criterios PRISMA. Procesa archivos RIS exportados desde Zotero.

## Requisitos
```bash
pip install -r requirements.txt
#INSTALAR (una sola vez en la terminal):
   # pip install rispy pandas numpy scikit-learn unidecode tqdm pycountry
```

## Uso
  1. Coloque su archivo `.ris` en esta carpeta
2. Edite `RIS_FILE` en la línea 44 del script
3. Ejecuta:
```bash
python screening_final.py
```

## Outputs generados (carpeta output/)
| Archivo | Contenido |
|---|---|
| 01_todos_los_articulos.csv | Base completa con etiquetas PRISMA |
| 02_prisma_flujo.csv | Conteos para diagrama PRISMA |
| 03_por_anio.csv | Tendencia temporal |
| 04_por_pais_continente.csv | Distribución geográfica |
| 05_por_revista.csv | Ranking de revistas |
| 06_revision_vs_investigacion.csv | Tipo de artículo |
| 07_antibioticos_adsorbentes.csv | Analito × adsorbente × autor |
| 08_variables_fisicoquimicas.csv | pHpzc, BET, pH op., matriz, regeneración |
| 09_datos_cuantitativos.csv | qmax, % remoción, isoterma, cinética |
| 10_bibliometrico.csv | Para análisis en R (bibliometrix, metafor) |

## Nota
El archivo `.ris` original no se incluye, debido al gran tamaño.
