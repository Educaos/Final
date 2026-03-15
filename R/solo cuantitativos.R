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

#Este es el pre
library(tidyverse)
library(viridis)
library(scales)

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