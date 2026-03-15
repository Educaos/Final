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

# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 1: (tendencia por año) 
# ══════════════════════════════════════════════════════════════════════════════

library(tidyverse)
library(ggplot2)

grafica_anio <- ggplot(por_anio, aes(x = year, y = n_articulos)) +
  geom_col(fill = "#CDB5CD", color = "white", width = 0.7) +
  geom_text(aes(label = n_articulos), vjust = -0.5, size = 3, color = "black") +
  scale_x_continuous(breaks = seq(2020, 2025, by = 1)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Evolución de publicaciones sobre adsorbentes para remoción de antibióticos y hormonas en medios acuosos",
    subtitle = paste("Total:", sum(por_anio$n_articulos), "artículos | Fuente: Web of Science & Scopus"),
    x = "Año de publicación",
    y = "Número de publicaciones",
    caption = "Distribución artpiculos por año| Metodología PRISMA | Trabajo de Grado"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  )

print(grafica_anio)
ggsave("graficas/01_distribucion_por_anio.png", plot = grafica_anio,
       width = 12, height = 7, dpi = 300)
cat("✅ Gráfica guardada!\n")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 2:   Top países y continentes
# ══════════════════════════════════════════════════════════════════════════════
# TOP 20 PAÍSES
top20 <- por_pais %>%
  arrange(desc(n_articulos)) %>%
  slice_head(n = 20)

grafica_paises <- ggplot(top20, aes(x = reorder(pais, n_articulos),
                                    y = n_articulos, fill = continente)) +
  geom_col() +
  geom_text(aes(label = paste0(n_articulos, " (", round(pct, 1), "%)")), 
            hjust = -0.2, size = 3.5) +
  coord_flip() +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title = "Top 20 países con mayor producción científica",
    subtitle = "Adsorbentes para remoción de antibióticos y hormonas en medios acuosos",
    x = NULL, y = "Número de publicaciones", fill = "Continente"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

print(grafica_paises)
ggsave("graficas/02_top20_paises.png", plot = grafica_paises,
       width = 10, height = 10, dpi = 300)

# POR CONTINENTE
por_continente <- por_pais %>%
  group_by(continente) %>%
  summarise(total = sum(n_articulos)) %>%
  arrange(desc(total)) %>%
  mutate(pct = round(total / sum(total) * 100, 1))
# ↑ porcentaje sobre el total global

grafica_continente <- ggplot(por_continente,
                             aes(x = reorder(continente, total),
                                 y = total, fill = continente)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(total, " (", pct, "%)")),
            hjust = -0.1, size = 4) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "Publicaciones por continente", x = NULL, y = "Número de publicaciones") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(grafica_continente)
ggsave("graficas/03_por_continente.png", plot = grafica_continente,
       width = 10, height = 6, dpi = 300)


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 3: top 15 revistas
# ══════════════════════════════════════════════════════════════════════════════

top15_revistas <- por_revista %>%
  arrange(desc(n_articulos)) %>%
  slice_head(n = 15)

grafica_revistas <- ggplot(top15_revistas,
                           aes(x = reorder(journal_raw, n_articulos), 
                               y = n_articulos)) +
  geom_col(fill = "#BC8F8F") +
  geom_text(aes(label = n_articulos), hjust = -0.2, size = 3.5) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Top 15 revistas científicas por número de publicaciones",
    subtitle = "Adsorbentes para remoción de antibióticos y hormonas en medios acuosos",
    x = NULL, y = "Número de artículos"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(grafica_revistas)
ggsave("graficas/04_top_revistas.png", plot = grafica_revistas,
       width = 12, height = 8, dpi = 300)
cat("✅ Gráfica de revistas guardada!\n")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 4:   Revisiones vs Investigación)
# ══════════════════════════════════════════════════════════════════════════════

resumen_tipo <- tipo_articulo %>%
  count(tipo_articulo) %>%
  rename(tipo = tipo_articulo, total = n)

grafica_tipo <- ggplot(resumen_tipo, aes(x = "", y = total, fill = tipo)) +
  geom_col(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(tipo, "\n", total, "\n(",
                               round(total/sum(total)*100, 1), "%)")),
            position = position_stack(vjust = 0.5), size = 4) +
  scale_fill_manual(values = c("#CDB5CD", "#8B7B8B", "#FF9800", "#4CAF50")) +
  labs(
    title = "Artículos de revisión vs. investigación original",
    subtitle = "Distribución de la literatura científica: artículos de investigación original vs. revisiones bibliográficas",
    fill = "Tipo"
  ) +
  theme_void(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

print(grafica_tipo)
ggsave("graficas/05_revision_vs_investigacion.png", plot = grafica_tipo,
       width = 8, height = 7, dpi = 300)
cat("✅ Gráfica de tipos guardada!\n")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 5 — Top antibióticos y top adsorbentes
# ══════════════════════════════════════════════════════════════════════════════

# TOP ANTIBIÓTICOS con NA explicado y porcentajes
top_analitos <- antibioticos %>%
  mutate(analitos = ifelse(is.na(analitos), "No especificado / Múltiples", analitos)) %>%
  count(analitos, sort = TRUE) %>%
  slice_head(n = 15) %>%
  mutate(
    pct = round(n / sum(n) * 100, 1),
    etiqueta = paste0(n, " (", pct, "%)")
  )

grafica_analitos <- ggplot(top_analitos, 
                           aes(x = reorder(analitos, n), y = n)) +
  geom_col(fill = "#8B7B8B") +
  geom_text(aes(label = etiqueta), hjust = -0.05, size = 3.5) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(
    title = "Antibióticos/Hormonas más estudiados",
    subtitle = "Top 15",
    x = NULL, 
    y = "Número de artículos",
    caption = "* 'No especificado / Múltiples': artículos que estudian distintos analitos\n O donde el analito no entra en las grandes categorías."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, color = "gray40", size = 9,
                                face = "italic"),
    plot.caption.position = "plot"
  )

print(grafica_analitos)
ggsave("graficas/06a_top_antibioticos.png", plot = grafica_analitos,
       width = 12, height = 8, dpi = 300)
cat("✅ Guardada con porcentajes y nota de NA!\n") 

# TOP ADSORBENTES
# TOP ADSORBENTES con NA explicado y porcentajes
top_adsorbentes <- antibioticos %>%
  mutate(adsorbente = ifelse(is.na(adsorbente), "No especificado / Múltiples", adsorbente)) %>%
  count(adsorbente, sort = TRUE) %>%
  slice_head(n = 15) %>%
  mutate(
    pct = round(n / sum(n) * 100, 1),
    etiqueta = paste0(n, " (", pct, "%)")
  )

grafica_adsorbentes <- ggplot(top_adsorbentes,
                              aes(x = reorder(adsorbente, n), y = n)) +
  geom_col(fill = "#CDB5CD") +
  geom_text(aes(label = etiqueta), hjust = -0.05, size = 3.5) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(
    title = "Adsorbentes más utilizados",
    subtitle = "Top 15",
    x = NULL,
    y = "Número de artículos",
    caption = "* 'No especificado / Múltiples': artículos que evalúan varios adsorbentes\n  artículos que estudian distintos analitos\n O donde el analito no entra en las grandes categorías."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0, color = "gray40", size = 9,
                                face = "italic"),
    plot.caption.position = "plot"
  )

print(grafica_adsorbentes)
ggsave("graficas/06b_top_adsorbentes.png", plot = grafica_adsorbentes,
       width = 12, height = 8, dpi = 300)
cat("✅ Guardada con porcentajes y nota de NA!\n")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 6:  Top adsorbentes por año
# ══════════════════════════════════════════════════════════════════════════════
library(tidyverse)
library(ggrepel)  # para etiquetas sin solapamiento

# ── 1. Identificar Top 10 adsorbentes globales ──────────────────────────────
top10_nombres <- antibioticos %>%
  filter(!is.na(adsorbente)) %>%
  count(adsorbente, sort = TRUE) %>%
  slice_head(n = 10) %>%
  pull(adsorbente)

# ── 2. Filtrar y contar por año ──────────────────────────────────────────────
ads_por_anio <- antibioticos %>%
  filter(adsorbente %in% top10_nombres, !is.na(year)) %>%
  count(year, adsorbente) %>%
  # Completar años sin datos con 0 (para que la línea no se corte)
  complete(year, adsorbente, fill = list(n = 0))

# ── 3. Etiquetas solo en el último año (para no saturar) ────────────────────
etiquetas_finales <- ads_por_anio %>%
  group_by(adsorbente) %>%
  filter(year == max(year)) %>%
  ungroup()

# ── 4. Gráfica ───────────────────────────────────────────────────────────────
grafica_lineas <- ggplot(ads_por_anio, 
                         aes(x = year, y = n, 
                             color = adsorbente, 
                             group = adsorbente)) +
  
  # Líneas
  geom_line(linewidth = 1, alpha = 0.85) +
  
  # Puntos
  geom_point(size = 2.5, alpha = 0.9) +
  
  # Etiquetas al final de cada línea
  geom_text_repel(
    data = etiquetas_finales,
    aes(label = adsorbente),
    nudge_x      = 0.5,
    hjust        = 0,
    size         = 3.2,
    fontface     = "bold",
    segment.size = 0.3,
    show.legend  = FALSE
  ) +
  
  scale_x_continuous(
    breaks = seq(min(ads_por_anio$year), max(ads_por_anio$year), by = 1),
    expand = expansion(mult = c(0.02, 0.25))
  ) + # espacio derecho para etiquetas
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_color_brewer(palette = "Paired") +
  
  labs(
    title    = "Evolución del uso de los Top 10 adsorbentes a través de los años",
    subtitle = "Número de publicaciones por año | Fuente: Web of Science & Scopus",
    x        = "Año de publicación",
    y        = "Número de artículos",
    color    = "Adsorbente",
    caption  = "Metodología PRISMA | Trabajo de Grado"
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5),
    plot.subtitle   = element_text(hjust = 0.5, color = "gray50"),
    plot.caption    = element_text(hjust = 0, color = "gray40", face = "italic"),
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "none",  # etiquetas directas en las líneas, no necesita leyenda
    panel.grid.minor = element_blank()
  )

print(grafica_lineas)

ggsave("graficas/08_top10_adsorbentes_por_anio.png",
       plot   = grafica_lineas,
       width  = 14, height = 8, dpi = 300)

cat("✅ Gráfica guardada en graficas/08_top10_adsorbentes_por_anio.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica 7: Mapa de calor de antibióticos vs adsorbente y hormonas vs adsorbente
# ══════════════════════════════════════════════════════════════════════════════

library(tidyverse)
library(viridis)

# ── CLASIFICACIÓN DE ANALITOS ─────────────────────────────────────────────────
hormonas_lista <- c("Estradiol (E2)", "Estrona (E1)", "Etinilestradiol (EE2)",
                    "Estriol (E3)", "Progesterona", "Androgenos")

antibioticos_lista <- c("Tetraciclina", "Ciprofloxacino", "Doxiciclina",
                        "Sulfametoxazol", "Ofloxacino", "Norfloxacino",
                        "Levofloxacino", "Amoxicilina", "Eritromicina",
                        "Trimetoprima", "Ampicilina", "Azitromicina",
                        "Cloranfenicol", "Clortetraciclina", "Oxitetraciclina")

# ── SEPARAR COMBINACIONES EN FILAS INDIVIDUALES ───────────────────────────────
datos_separados <- antibioticos %>%
  filter(!is.na(analitos), !is.na(adsorbente)) %>%
  mutate(adsorbente = ifelse(is.na(adsorbente), "No especificado", adsorbente)) %>%
  separate_rows(analitos, sep = "; ") %>%       # separa "A; B; C" en 3 filas
  mutate(analitos = str_trim(analitos)) %>%      # elimina espacios extra
  mutate(tipo = case_when(
    analitos %in% hormonas_lista    ~ "hormona",
    analitos %in% antibioticos_lista ~ "antibiotico",
    TRUE ~ "otro"
  ))

# ── Top 10 adsorbentes globales ───────────────────────────────────────────────
top10_ads <- datos_separados %>%
  count(adsorbente, sort = TRUE) %>%
  slice_head(n = 10) %>%
  pull(adsorbente)

# ══════════════════════════════════════════════════════════════════════════════
# MAPA DE CALOR 1 — ANTIBIÓTICOS vs ADSORBENTE
# ══════════════════════════════════════════════════════════════════════════════
mapa_anti <- datos_separados %>%
  filter(tipo == "antibiotico", adsorbente %in% top10_ads) %>%
  count(analitos, adsorbente) %>%
  # Completar combinaciones sin datos con 0
  complete(analitos, adsorbente, fill = list(n = 0))

grafica_mapa_anti <- ggplot(mapa_anti,
                            aes(x = adsorbente, y = analitos, fill = n)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = ifelse(n > 0, n, "")),
            size = 3.2, color = "white", fontface = "bold") +
  scale_fill_viridis_c(
    option = "plasma",
    name   = "N° artículos",
    breaks = pretty,
    labels = scales::comma
  ) +
  labs(
    title    = "Antibióticos estudiados vs Adsorbente utilizado",
    subtitle = "Top 10 adsorbentes más frecuentes | Celdas vacías = sin combinación reportada",
    x        = "Adsorbente",
    y        = "Antibiótico",
    caption  = "Fuente: Web of Science & Scopus | Metodología PRISMA"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y   = element_text(size = 10),
    panel.grid    = element_blank(),
    legend.position = "right"
  )

print(grafica_mapa_anti)
ggsave("graficas/07a_mapa_calor_antibioticos.png",
       plot  = grafica_mapa_anti,
       width = 14, height = 10, dpi = 300)
cat("✅ Mapa de calor antibióticos guardado!\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAPA DE CALOR 2 — HORMONAS vs ADSORBENTE
# ══════════════════════════════════════════════════════════════════════════════
mapa_horm <- datos_separados %>%
  filter(tipo == "hormona", adsorbente %in% top10_ads) %>%
  count(analitos, adsorbente) %>%
  complete(analitos, adsorbente, fill = list(n = 0))

grafica_mapa_horm <- ggplot(mapa_horm,
                            aes(x = adsorbente, y = analitos, fill = n)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = ifelse(n > 0, n, "")),
            size = 3.5, color = "white", fontface = "bold") +
  scale_fill_viridis_c(
    option = "mako",        # color diferente para distinguirlo del de antibióticos
    name   = "N° artículos",
    breaks = pretty,
    labels = scales::comma
  ) +
  labs(
    title    = "Hormonas estudiadas vs Adsorbente utilizado",
    subtitle = "Top 10 adsorbentes más frecuentes | Celdas vacías = sin combinación reportada",
    x        = "Adsorbente",
    y        = "Hormona",
    caption  = "Fuente: Web of Science & Scopus | Metodología PRISMA"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y   = element_text(size = 10),
    panel.grid    = element_blank(),
    legend.position = "right"
  )

print(grafica_mapa_horm)
ggsave("graficas/07b_mapa_calor_hormonas.png",
       plot  = grafica_mapa_horm,
       width = 12, height = 8, dpi = 300)
cat("✅ Mapa de calor hormonas guardado!\n")









# ── Isotermas por adsorbente ──────────────────────────────────────────────────
isoterma_data <- cuantitativos %>%
  filter(!is.na(isoterma), !is.na(adsorbente)) %>%
  count(adsorbente, isoterma) %>%
  group_by(adsorbente) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  ungroup()

grafica_isoterma <- ggplot(isoterma_data,
                            aes(x = adsorbente, y = pct, fill = isoterma)) +
  geom_col(position = "stack", color = "white", linewidth = 0.4) +
  geom_text(aes(label = ifelse(pct >= 8, paste0(pct, "%"), "")),
            position = position_stack(vjust = 0.5),
            size = 3.2, color = "white", fontface = "bold") +
  scale_fill_viridis_d(option = "plasma", name = "Modelo de isoterma") +
  scale_y_continuous(labels = percent_format(scale = 1),
                     expand = expansion(mult = c(0, 0.03))) +
  labs(
    title    = "Modelos de isoterma reportados por tipo de adsorbente",
    subtitle = paste0("n = ", sum(isoterma_data$n), " artículos con isoterma reportada"),
    x        = "Tipo de adsorbente",
    y        = "Proporción (%)",
    caption  = "Langmuir = adsorción en monocapa | Freundlich = superficie heterogénea\nSips = modelo mixto | Temkin = incluye interacciones adsorbato-adsorbente"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic", size = 9),
    axis.text.x   = element_text(angle = 30, hjust = 1),
    panel.grid.major.x = element_blank()
  )

print(grafica_isoterma)
ggsave("graficas/11a_isotermas_por_adsorbente.png",
       plot = grafica_isoterma, width = 13, height = 8, dpi = 300)
cat("✅ Isotermas guardada!\n")


#CUANTITATIVOS
# Cargar datos
cuantitativos <- read_csv("data/09_datos_cuantitativos_4674.csv")

# Limpiar adsorbente
cuantitativos <- cuantitativos %>%
  mutate(adsorbente = ifelse(is.na(adsorbente), "No especificado", adsorbente))


# ══════════════════════════════════════════════════════════════════════════════
# Gráfica A:  % Remoción por adsorbente (Boxplot) 
# ════════════════════════════════════════════════════════════════════════
remocion_data <- cuantitativos %>%
  filter(!is.na(remocion_pct), adsorbente != "No especificado") %>%
  group_by(adsorbente) %>%
  filter(n() >= 5) %>%   # mínimo 5 datos por adsorbente
  ungroup()

# Calcular medianas para ordenar
medianas <- remocion_data %>%
  group_by(adsorbente) %>%
  summarise(
    mediana = median(remocion_pct),
    n       = n(),
    .groups = "drop"
  )

remocion_data <- remocion_data %>%
  left_join(medianas, by = "adsorbente")

grafica_remocion <- ggplot(remocion_data,
                           aes(x = reorder(adsorbente, mediana),
                               y = remocion_pct,
                               fill = adsorbente)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.85,
               outlier.shape = 21, outlier.size = 1.5,
               outlier.alpha = 0.5) +
  geom_hline(yintercept = 90, linetype = "dashed",
             color = "red", linewidth = 0.8) +
  annotate("text", x = 0.6, y = 91.5,
           label = "90% remoción", color = "red",
           size = 3.5, hjust = 0) +
  # Mostrar n de cada grupo
  geom_text(data = medianas %>% filter(n >= 5),
            aes(x = adsorbente, y = -3,
                label = paste0("n=", n)),
            size = 3, color = "gray40", inherit.aes = FALSE) +
  coord_flip() +
  scale_fill_viridis_d(option = "magma") +
  scale_y_continuous(limits = c(-5, 105),
                     breaks = seq(0, 100, by = 20)) +
  labs(
    title    = "Eficiencia de remoción (%) por tipo de adsorbente",
    subtitle = "Solo adsorbentes con ≥ 5 datos reportados | Línea roja = 90% remoción",
    x        = NULL,
    y        = "% Remoción",
    caption  = "Caja = percentiles 25–75% | Línea central = mediana | Puntos = valores atípicos"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    panel.grid.major.y = element_blank()
  )

print(grafica_remocion)
ggsave("graficas/10a_remocion_por_adsorbente.png",
       plot = grafica_remocion, width = 12, height = 8, dpi = 300)
cat("✅ Gráfica A guardada!\n")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA B — Isotermas por adsorbente (Barras apiladas %)
# ════════════════════════════════════════════════════════════════════════
#Muestra qué modelo de adsorción predomina en cada material

isoterma_data <- cuantitativos %>%
  filter(!is.na(isoterma), adsorbente != "No especificado") %>%
  # Simplificar isotermas combinadas
  mutate(isoterma_simple = case_when(
    str_detect(isoterma, "Langmuir") & str_detect(isoterma, "Freundlich") ~ "Langmuir + Freundlich",
    str_detect(isoterma, "Langmuir")   ~ "Langmuir",
    str_detect(isoterma, "Freundlich") ~ "Freundlich",
    str_detect(isoterma, "Sips")       ~ "Sips",
    str_detect(isoterma, "Temkin")     ~ "Temkin",
    TRUE ~ "Otra"
  )) %>%
  group_by(adsorbente) %>%
  filter(n() >= 10) %>%
  ungroup() %>%
  count(adsorbente, isoterma_simple) %>%
  group_by(adsorbente) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

grafica_isoterma <- ggplot(isoterma_data,
                           aes(x = reorder(adsorbente, pct),
                               y = pct,
                               fill = isoterma_simple)) +
  geom_col(position = "stack", color = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(pct >= 8, paste0(round(pct), "%"), "")),
            position = position_stack(vjust = 0.5),
            size = 3, color = "white", fontface = "bold") +
  coord_flip() +
  scale_fill_viridis_d(option = "magma", name = "Modelo de isoterma") +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title    = "Modelos de isoterma de adsorción por tipo de adsorbente",
    subtitle = "Distribución porcentual | Solo adsorbentes con ≥ 10 datos | Muestra qué modelo de adsorción predomina en cada material

",
    x        = NULL,
    y        = "% de artículos",
    caption  = "Langmuir = adsorción en monocapa | Freundlich = superficie heterogénea | Sips = modelo mixto"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

print(grafica_isoterma)
ggsave("graficas/10b_isotermas_por_adsorbente.png",
       plot = grafica_isoterma, width = 13, height = 8, dpi = 300)
cat("✅ Gráfica B guardada!\n")



# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA C — Cinética por adsorbente (Barras apiladas %)
# ════════════════════════════════════════════════════════════════════════
# Muestra si la adsorción es rápida (PFO) o controlada (PSO)

cinetica_data <- cuantitativos %>%
  filter(!is.na(cinetica), adsorbente != "No especificado") %>%
  mutate(cinetica_simple = case_when(
    str_detect(cinetica, "PSO") & str_detect(cinetica, "PFO") ~ "PFO + PSO",
    str_detect(cinetica, "PSO") ~ "PSO (pseudo-segundo orden)",
    str_detect(cinetica, "PFO") ~ "PFO (pseudo-primer orden)",
    TRUE ~ "Otro"
  )) %>%
  group_by(adsorbente) %>%
  filter(n() >= 10) %>%
  ungroup() %>%
  count(adsorbente, cinetica_simple) %>%
  group_by(adsorbente) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

grafica_cinetica <- ggplot(cinetica_data,
                           aes(x = reorder(adsorbente, pct),
                               y = pct,
                               fill = cinetica_simple)) +
  geom_col(position = "stack", color = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(pct >= 8, paste0(round(pct), "%"), "")),
            position = position_stack(vjust = 0.5),
            size = 3.2, color = "white", fontface = "bold") +
  coord_flip() +
  scale_fill_manual(
    name   = "Modelo cinético",
    values = c("PFO (pseudo-primer orden)"  = "#CDB5CD",
               "PSO (pseudo-segundo orden)" = "#8B7B8B",
               "PFO + PSO"                  = "#524552",
               "Otro"                       = "#9E9E9E")
  ) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title    = "Modelos cinéticos de adsorción por tipo de adsorbente",
    subtitle = "Distribución porcentual | Solo adsorbentes con ≥ 10 datos",
    x        = NULL,
    y        = "% de artículos",
    caption  = "PSO = pseudo-segundo orden (más común, adsorción química)\nPFO = pseudo-primer orden (adsorción física)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

print(grafica_cinetica)
ggsave("graficas/10c_cinetica_por_adsorbente.png",
       plot = grafica_cinetica, width = 13, height = 8, dpi = 300)
cat("✅ Gráfica C guardada!\n")



# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA D — Matriz real vs sintética por adsorbente
# ════════════════════════════════════════════════════════════════════════
# Importante para evaluar aplicabilidad real del adsorbente

matriz_data <- cuantitativos %>%
  filter(!is.na(tipo_matriz), adsorbente != "No especificado") %>%
  group_by(adsorbente) %>%
  filter(n() >= 5) %>%
  ungroup() %>%
  count(adsorbente, tipo_matriz) %>%
  group_by(adsorbente) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

grafica_matriz <- ggplot(matriz_data,
                         aes(x = reorder(adsorbente, pct),
                             y = pct, fill = tipo_matriz)) +
  geom_col(position = "stack", color = "white") +
  geom_text(aes(label = paste0(round(pct), "%")),
            position = position_stack(vjust = 0.5),
            size = 3.5, color = "white", fontface = "bold") +
  coord_flip() +
  scale_fill_manual(
    name   = "Tipo de matriz",
    values = c("Sintetica" = "#CDB5CD", "Real" = "#8B7B8B")
  ) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title    = "Tipo de matriz ensayada por adsorbente",
    subtitle = "Matriz real vs. sintética | Importante para evaluar aplicabilidad",
    x        = NULL,
    y        = "% de estudios",
    caption  = "Matriz real = agua residual real | Matriz sintética = solución preparada en laboratorio"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "gray40", face = "italic"),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

print(grafica_matriz)
ggsave("graficas/10d_matriz_por_adsorbente.png",
       plot = grafica_matriz, width = 12, height = 8, dpi = 300)
cat("✅ Gráfica D guardada!\n")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA E — BET por adsorbente (los 189 datos disponibles)
# ════════════════════════════════════════════════════════════════════════

bet_data <- cuantitativos %>%
  filter(!is.na(bet_m2g), adsorbente != "No especificado") %>%
  group_by(adsorbente) %>%
  filter(n() >= 3) %>%
  ungroup()

grafica_bet <- ggplot(bet_data,
                      aes(x = reorder(adsorbente, bet_m2g, median),
                          y = bet_m2g, fill = adsorbente)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.85) +
  scale_y_log10(labels = comma) +
  coord_flip() +
  scale_fill_viridis_d(option = "plasma") +
  labs(
    title    = "Área superficial BET por tipo de adsorbente",
    subtitle = "Escala logarítmica | Solo adsorbentes con ≥ 3 datos (n = 189 total)",
    x        = NULL,
    y        = "Área BET (m²/g) — escala log",
    caption  = "⚠️ Datos limitados (n=189). Mayor BET generalmente indica mayor capacidad de adsorción."
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 10),
    plot.caption  = element_text(hjust = 0, color = "orange3", face = "italic"),
    panel.grid.major.y = element_blank()
  )

print(grafica_bet)
ggsave("graficas/10e_BET_por_adsorbente.png",
       plot = grafica_bet, width = 12, height = 7, dpi = 300)
cat("✅ Gráfica E guardada!\n")
