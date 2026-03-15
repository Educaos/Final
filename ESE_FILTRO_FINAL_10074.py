# -*- coding: utf-8 -*-
"""
=============================================================================
SCREENING BIBLIOGRÁFICO — REVISIÓN SISTEMÁTICA CON METAANÁLISIS
Bioadsorbentes frente a antibióticos y hormonas en medios acuosos
=============================================================================

INSTALAR (una sola vez en la terminal):
    pip install rispy pandas numpy scikit-learn unidecode tqdm pycountry

EJECUTAR:
#Ejecutar Documento final de filtrado.
    python screening_final.py

OUTPUTS en carpeta "output/":
    01_todos_los_articulos.csv       → Base completa con etiqueta y razón
    02_prisma_flujo.csv              → Conteos para diagrama PRISMA
    03_por_anio.csv                  → Publicaciones por año
    04_por_pais_continente.csv       → País y continente
    05_por_revista.csv               → Ranking de revistas
    06_revision_vs_investigacion.csv → Revisiones vs originales
    07_antibioticos_adsorbentes.csv  → Analito + adsorbente + autor
    08_variables_fisicoquimicas.csv  → pHpzc, BET, pH op., dosis, matriz...
    09_datos_cuantitativos.csv       → qmax, remoción, isoterma, cinética
    10_bibliometrico.csv             → Todo junto para bibliometrix en R
=============================================================================
"""

import os, re, sys, time, unicodedata, warnings
import numpy as np
import pandas as pd
import pycountry
import rispy
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN  ← CAMBIA SOLO ESTA LÍNEA
# ─────────────────────────────────────────────────────────────────────────────
RIS_FILE   = "F_biblio_zotero_29131_afterclean.ris"
OUTPUT_DIR = "output"
YEAR_MIN   = 2020
YEAR_MAX   = 2025
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()

def seg():
    return f"[{time.time()-t0:>6.1f}s]"

def titulo_seccion(txt):
    print("\n" + "=" * 65)
    print(f"  {txt}")
    print("=" * 65)

titulo_seccion("SCREENING — BIOADSORBENTES / CONTAMINANTES EMERGENTES")

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN (única función en todo el script)
# ─────────────────────────────────────────────────────────────────────────────
def norm(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, list):
        x = " ".join(str(i) for i in x if i is not None)
    x = str(x)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", x.lower().strip())

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — CARGAR RIS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 1/9 — Cargando archivo RIS...")

if not os.path.exists(RIS_FILE):
    print(f"  ERROR: No se encontro '{RIS_FILE}'")
    print(f"  Carpeta actual: {os.getcwd()}")
    sys.exit(1)

with open(RIS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    registros = rispy.load(f)

df = pd.DataFrame(registros)
total_cargados = len(df)
print(f"  OK  Registros cargados : {total_cargados:,}")
print(f"      Columnas del RIS   : {list(df.columns[:12])} ...")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — NORMALIZAR COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 2/9 — Normalizando columnas...")

def get_col(df_, *nombres):
    for n in nombres:
        if n in df_.columns:
            return df_[n].fillna("")
    return pd.Series([""] * len(df_), index=df_.index)

df["title_raw"]    = get_col(df, "title", "primary_title")
df["abstract_raw"] = get_col(df, "abstract", "notes")
df["journal_raw"]  = get_col(df, "journal_name", "journal", "secondary_title", "alternate_title3")
df["doi_raw"]      = get_col(df, "doi")
df["authors_raw"]  = get_col(df, "authors", "author")
df["type_raw"]     = get_col(df, "type_of_reference", "type", "TY")
df["keywords_raw"] = get_col(df, "keywords", "keyword")
df["db_raw"]       = get_col(df, "database", "db", "data_source", "source")

df["autor1"] = df["authors_raw"].apply(
    lambda x: norm(x[0]) if isinstance(x, list) and len(x) > 0 else norm(x)
)

def extraer_anio(fila):
    for campo in ["year", "publication_year", "date", "py"]:
        if campo in fila.index and pd.notna(fila[campo]):
            m = re.search(r"(19|20)\d{2}", str(fila[campo]))
            if m:
                return int(m.group())
    return None

df["year"]     = df.apply(extraer_anio, axis=1)
df["doi_norm"] = df["doi_raw"].apply(norm)
df["texto"]    = (df["title_raw"].apply(norm) + " " + df["abstract_raw"].apply(norm)).str.strip()

print(f"  OK  Columnas normalizadas")
print(f"      Titulo ejemplo : {str(df['title_raw'].iloc[0])[:70]}...")
print(f"      Anios con dato : {df['year'].notna().sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — DEDUPLICACIÓN RÁPIDA (Frecuecnia de termino-IDF)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 3/9 — Detectando duplicados (puede tardar 1-3 min)...")

# 3A — DOI exacto
tiene_doi   = df["doi_norm"].str.len() > 0
df["dup_doi"] = False
df.loc[tiene_doi, "dup_doi"] = df.loc[tiene_doi].duplicated(subset=["doi_norm"], keep="first")
n_dup_doi = int(df["dup_doi"].sum())
print(f"  OK  Duplicados por DOI exacto    : {n_dup_doi:,}")

# 3B — Titulo similar (TF-IDF coseno — 100x más rápido que loop fuzzy)
df["dup_fuzzy"] = False
n_dup_fuzzy   = 0
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cand = df[~df["dup_doi"]].copy()
    cand["year_bucket"]   = cand["year"].fillna(-1).astype(int)
    cand["titulo_corto"]  = df["title_raw"].apply(lambda x: " ".join(norm(x).split()[:12]))

    dup_idx = set()
    for ano in tqdm(sorted(cand["year_bucket"].unique()),
                    desc="      Comparando titulos", ncols=65):
        mask   = cand["year_bucket"].isin([ano - 1, ano, ano + 1])
        grupo  = cand[mask]
        if len(grupo) < 2:
            continue
        titulos = grupo["titulo_corto"].tolist()
        indices = grupo.index.tolist()
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
            mat = vec.fit_transform(titulos)
            sim = cosine_similarity(mat)
        except Exception:
            continue
        marcado = set()
        for i in range(len(indices)):
            if indices[i] in marcado:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in marcado:
                    continue
                if sim[i, j] >= 0.85:
                    dup_idx.add(indices[j])
                    marcado.add(indices[j])

    df.loc[list(dup_idx), "dup_fuzzy"] = True
    n_dup_fuzzy = int(df["dup_fuzzy"].sum())
    print(f"  OK  Duplicados por titulo similar : {n_dup_fuzzy:,}")

except ImportError:
    print("  AVISO: Instala scikit-learn:  pip install scikit-learn")

df_base = df[~(df["dup_doi"] | df["dup_fuzzy"])].copy()
print(f"  OK  Registros unicos             : {len(df_base):,}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — KEYWORDS, SCORING Y CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 4/9 — Aplicando criterios de inclusion/exclusion...")

KW_OBJETIVO = [
    "antibiotic","antibiotics","antibiotico","antibioticos",
    "pharmaceutical","pharmaceuticals","farmaceutico",
    "emerging contaminant","emerging contaminants",
    "contaminante emergente","contaminantes emergentes",
    "micropollutant","micropollutants","micropoluente",
    "fluoroquinolone","fluoroquinolones","quinolone","quinolones",
    "sulfonamide","sulfonamides","macrolide","macrolides",
    "beta-lactam","betalactam","cephalosporin","cephalosporins",
    "tetracycline","oxytetracycline","chlortetracycline","doxycycline",
    "tetraciclina","oxitetraciclina","clortetraciclina","doxiciclina",
    "ciprofloxacin","norfloxacin","ofloxacin","levofloxacin",
    "ciprofloxacino","norfloxacino","ofloxacino",
    "sulfamethoxazole","trimethoprim","sulfametoxazol",
    "amoxicillin","ampicillin","amoxicilina","ampicilina",
    "chloramphenicol","cloranfenicol",
    "erythromycin","azithromycin","clarithromycin",
    "eritromicina","azitromicina","claritromicina",
    "hormone","hormones","hormona","hormonas",
    "estrogen","estrogens","estrogeno","estrona","estradiol",
    "ethinylestradiol","estriol","17beta-estradiol",
    "progesterone","progesterona","progestin","progestogen",
    "androgen","androgens","androgeno",
    "endocrine disrupting","endocrine disruptor",
    "disruptor endocrino","disruptores endocrinos",
]
KW_AGUA = [
    "water","wastewater","waste water","agua","aguas",
    "agua residual","aguas residuales","efluente","solucion acuosa",
    "medio acuoso","effluent","aqueous","aqueous solution",
    "surface water","groundwater","river water","lake water",
    "drinking water","sewage","wwtp","wastewater treatment plant",
    "industrial wastewater","hospital wastewater","municipal wastewater",
    "synthetic wastewater","tratamiento de aguas",
]
KW_ADSORCION = [
    "adsorption","adsorbent","adsorcion","adsorbente",
    "sorption","biosorption","biosorcion",
    "removal","remocion","elimination","eliminacion",
    "biochar","activated carbon","carbon activado",
    "mof","metal-organic framework","metal organic framework",
    "nanocomposite","nanoparticle","nanomaterial",
    "biopolymer","chitosan","quitosano","alginate","alginato",
    "cellulose","celulosa","lignin","lignina",
    "zeolite","zeolita","clay","bentonite",
    "qmax","qe","mg/g","mg g-1",
    "adsorption capacity","capacidad de adsorcion",
    "removal efficiency","percentage removal","porcentaje de remocion",
    "langmuir","freundlich","temkin",
    "isotherm","isoterma","kinetic","kinetics","cinetica",
    "pseudo-first-order","pseudo second order","pfo","pso",
    "surface area","area superficial","pore size","pore volume",
    "ftir","xps","functional group","grupos funcionales",
    "regeneration","reusability","desorption",
]
KW_EXCL_CONTAM = [
    "pfas","perfluoro","perfluorinated",
    # pesticidas — solo si son el foco PRINCIPAL
    "pesticide removal","pesticides adsorption","herbicide removal",
    # metales pesados — solo si son el foco, no si son dopantes
    "heavy metal removal","lead removal","cadmium removal",
    "mercury removal","arsenic removal","chromium removal",
    # microplásticos
    "microplastic removal","microplastics adsorption",
    # colorantes — muy frecuente en adsorción pero fuera de foco
    "methylene blue removal","rhodamine removal",
    "dye removal","dye adsorption",
    # petróleo
    "oil spill","crude oil removal",
]
KW_EXCL_ESTUDIO = [
    # sensores — solo detección sin remoción
    "electrochemical sensor","optical sensor","fluorescent sensor",
    "biosensor for detection","colorimetric sensor",
    # biomédico puro
    "drug delivery","nanocarrier","controlled release",
    "pharmacokinetic","clinical trial","animal model",
    "cell line","in vitro cytotoxicity",
    # genética / resistencia antimicrobiana (diferente a remoción)
    "antimicrobial resistance gene","16s rrna","qpcr","integron",
    # degradación sin adsorción
    "photocatalytic degradation only","fenton degradation",
    "electrochemical oxidation","ozonation",
]
KW_EXCL_DOC = [
    "conference","proceedings","conference paper",
    "editorial","letter","short communication",
    "erratum","corrigendum","retracted","retraction",
    "patent","preprint","biorxiv",
]

def make_rx(lista):
    esc = [re.escape(norm(k)) for k in lista if k.strip()]
    return re.compile(r"(" + "|".join(esc) + r")", re.IGNORECASE)

RX_OBJ   = make_rx(KW_OBJETIVO)
RX_AGU   = make_rx(KW_AGUA)
RX_ADS   = make_rx(KW_ADSORCION)
RX_EXC_C = make_rx(KW_EXCL_CONTAM)
RX_EXC_E = make_rx(KW_EXCL_ESTUDIO)
RX_EXC_D = make_rx(KW_EXCL_DOC)

def hits(r, t): return len(r.findall(t)) if t else 0
def flag(r, t): return 1 if (t and r.search(t)) else 0

print("      Contando coincidencias...")
df_base["hits_obj"] = df_base["texto"].apply(lambda t: hits(RX_OBJ, t))
df_base["hits_agu"] = df_base["texto"].apply(lambda t: hits(RX_AGU, t))
df_base["hits_ads"] = df_base["texto"].apply(lambda t: hits(RX_ADS, t))
df_base["exc_c"]    = df_base["texto"].apply(lambda t: flag(RX_EXC_C, t))
df_base["exc_e"]    = df_base["texto"].apply(lambda t: flag(RX_EXC_E, t))
df_base["exc_d"]    = df_base["texto"].apply(lambda t: flag(RX_EXC_D, t))
df_base["anio_ok"]  = df_base["year"].between(YEAR_MIN, YEAR_MAX, inclusive="both").fillna(False)

df_base["score"] = (
    df_base["hits_obj"] * 4 + df_base["hits_agu"] * 2 + df_base["hits_ads"] * 3
    - df_base["exc_c"] * 8 - df_base["exc_e"] * 6 - df_base["exc_d"] * 6
)
df_base["pasa_core"] = (
    (df_base["hits_obj"] > 0) & (df_base["hits_agu"] > 0) &
    (df_base["hits_ads"] > 0) & (df_base["anio_ok"] == True)
)

def razones(r):
    R = []
    if not r["anio_ok"]:     R.append(f"EXCL: anio fuera de {YEAR_MIN}-{YEAR_MAX}")
    if r["exc_d"]:           R.append("EXCL: tipo documental (conferencia/patente/erratum)")
    if r["exc_c"]:           R.append("EXCL: contaminante fuera de foco (pesticida/metal/colorante)")
    if r["exc_e"]:           R.append("EXCL: tipo de estudio (sensor/biomedico/genetica/AOP)")
    if r["hits_obj"] == 0:   R.append("EXCL: no menciona antibiotico ni hormona")
    if r["hits_agu"] == 0:   R.append("EXCL: no menciona medio acuoso")
    if r["hits_ads"] == 0:   R.append("EXCL: no menciona adsorcion ni adsorbente")
    if not R and r["pasa_core"]: R.append("INCLUIDO: cumple todos los criterios")
    return " | ".join(R) if R else "REVISAR MANUALMENTE"

def etiquetar(r):
    if r["exc_d"] or r["exc_c"]:            return "EXCLUIDO"
    if r["exc_e"] and r["score"] <= 2:      return "EXCLUIDO"
    if not r["anio_ok"]:                    return "EXCLUIDO"
    if r["pasa_core"] and r["score"] >= 12: return "INCLUIDO_ALTA"
    if r["pasa_core"] and r["score"] >= 7:  return "INCLUIDO_MEDIA"
    if r["score"] >= 3:                     return "PRIORIDAD_BAJA"
    return "EXCLUIDO"

df_base["etiqueta"] = df_base.apply(etiquetar, axis=1)
df_base["razones"]  = df_base.apply(razones,   axis=1)

dist = df_base["etiqueta"].value_counts()
mx   = dist.max()
print("  OK  Clasificacion completada:")
for etiq, n in dist.items():
    barra = "=" * (n * 28 // mx)
    print(f"      {etiq:<22} {n:>7,}  {barra}")

df_inc = df_base[df_base["etiqueta"].isin(["INCLUIDO_ALTA", "INCLUIDO_MEDIA"])].copy()
print(f"\n      >>> Articulos incluidos (alta + media): {len(df_inc):,} <<<")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — PAÍS Y CONTINENTE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 5/9 — Detectando pais y continente...")

CONTINENTE = {
    "united states":"America del Norte","canada":"America del Norte","mexico":"America del Norte",
    "brazil":"America del Sur","colombia":"America del Sur","argentina":"America del Sur",
    "chile":"America del Sur","peru":"America del Sur","venezuela":"America del Sur",
    "ecuador":"America del Sur","bolivia":"America del Sur","paraguay":"America del Sur",
    "uruguay":"America del Sur","cuba":"America Central/Caribe","costa rica":"America Central/Caribe",
    "germany":"Europa","france":"Europa","spain":"Europa","italy":"Europa",
    "united kingdom":"Europa","poland":"Europa","netherlands":"Europa","sweden":"Europa",
    "portugal":"Europa","greece":"Europa","czech republic":"Europa","belgium":"Europa",
    "romania":"Europa","hungary":"Europa","denmark":"Europa","switzerland":"Europa",
    "austria":"Europa","norway":"Europa","finland":"Europa","turkey":"Europa",
    "china":"Asia","india":"Asia","iran":"Asia","korea republic of":"Asia",
    "japan":"Asia","pakistan":"Asia","saudi arabia":"Asia","taiwan":"Asia",
    "thailand":"Asia","malaysia":"Asia","indonesia":"Asia","viet nam":"Asia",
    "bangladesh":"Asia","sri lanka":"Asia","iraq":"Asia","jordan":"Asia",
    "egypt":"Africa","nigeria":"Africa","south africa":"Africa","morocco":"Africa",
    "ethiopia":"Africa","ghana":"Africa","kenya":"Africa","algeria":"Africa",
    "australia":"Oceania","new zealand":"Oceania",
}

lista_paises = list(
    {norm(c.name) for c in pycountry.countries} |
    {norm(c.official_name) for c in pycountry.countries if hasattr(c, "official_name")}
)

ALIAS_P = {
    "usa":"united states","u.s.a":"united states","u.s":"united states",
    "uk":"united kingdom","england":"united kingdom",
    "scotland":"united kingdom","wales":"united kingdom",
    "brasil":"brazil","espana":"spain",
    "pr china":"china","peoples republic of china":"china",
    "south korea":"korea republic of","vietnam":"viet nam",
}

def detectar_pais(texto):
    t = norm(texto)
    if not t: return None
    for a, e in ALIAS_P.items():
        t = t.replace(a, e)
    encontrados = [p for p in lista_paises
                   if re.search(r"\b" + re.escape(p) + r"\b", t)]
    if not encontrados:
        return None
    # devuelve el país que aparece más veces (primer autor generalmente)
    return max(set(encontrados), key=encontrados.count)

def txt_afil(fila):
    partes = []
    for col in ["affiliations","addresses","address","notes","journal_name"]:
        if col in fila.index and fila[col] is not None:
            v = fila[col]
            if isinstance(v, list): v = " ".join(str(x) for x in v)
            partes.append(str(v))
    return " | ".join(partes)

print("      Detectando pais por afiliacion...")
df_base["texto_afil"] = df_base.apply(txt_afil, axis=1)
df_base["pais"]       = df_base["texto_afil"].apply(detectar_pais)
df_base["continente"] = df_base["pais"].map(CONTINENTE).fillna("Sin determinar")

df_inc["texto_afil"] = df_inc.apply(txt_afil, axis=1)
df_inc["pais"]       = df_inc["texto_afil"].apply(detectar_pais)
df_inc["continente"] = df_inc["pais"].map(CONTINENTE).fillna("Sin determinar")

print(f"  OK  Pais detectado en: {df_inc['pais'].notna().sum():,} articulos incluidos")
print("      Top 10 paises:")
for p, n in df_inc["pais"].value_counts().head(10).items():
    cont = CONTINENTE.get(str(p), "?")
    print(f"        {str(p):<30} {n:>5}  ({cont})")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — FAMILIA DE ADSORBENTE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 6/9 — Detectando familia de adsorbente...")

FAMILIAS = {
    "Carbon activado":   ["activated carbon","carbon activado","granular activated carbon","powdered activated carbon","pac","gac"],
    "Biochar":           ["biochar","biocarbon","char","carbonized biomass","pirolisis","pirolizado"],
    "MOF":               ["mof","metal-organic framework","metal organic framework","zeolitic imidazolate","zif"],
    "Nanocompuesto":     ["nanocomposite","nano-composite","nanoparticle","nanomaterial","graphene oxide","carbon nanotube","cnt"],
    "Biopolimero":       ["biopolymer","chitosan","quitosano","alginate","alginato","cellulose","celulosa","starch","almidon","lignin","lignina"],
    "Arcilla/Zeolita":   ["clay","zeolite","zeolita","bentonite","montmorillonite"],
    "Hidrogel/Polimero": ["hydrogel","polymer","polimero","resin","resina"],
}

def detect_ads(t):
    for fam, kws in FAMILIAS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(norm(kw)) + r"\b", t):
                return fam
    return None

df_inc["adsorbente"] = df_inc["texto"].apply(detect_ads)
print("  OK  Adsorbentes en articulos incluidos:")
for a, n in df_inc["adsorbente"].value_counts().items():
    print(f"        {str(a):<22} : {n:>5}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — REVISIÓN vs INVESTIGACIÓN ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 7/9 — Clasificando revision vs investigacion original...")

KW_REVIEW = [
    "review","systematic review","meta-analysis","meta analysis","metaanalysis",
    "revision sistematica","literatura revisada","overview",
    "state of the art","scoping review","bibliometric",
]
RX_REVIEW = make_rx(KW_REVIEW)

def es_revision(fila):
    tipo = norm(str(fila.get("type_raw", "")))
    if any(r in tipo for r in ["review","rev"]):
        return "Revision"
    if RX_REVIEW.search(fila.get("texto", "")):
        return "Revision"
    return "Investigacion original"

df_inc["tipo_articulo"] = df_inc.apply(es_revision, axis=1)
rev_dist = df_inc["tipo_articulo"].value_counts()
print("  OK  Tipo de articulo:")
for t, n in rev_dist.items():
    print(f"        {t:<28} : {n:>5}")
print("      (Revisiones = tus ANTECEDENTES para la introduccion)")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 8 — ANALITOS + VARIABLES FISICOQUÍMICAS + DATOS CUANTITATIVOS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 8/9 — Extrayendo analitos y variables (fisicoquimicas + cuantitativas)...")

# Analitos específicos
ANALITOS = {
    "Tetraciclina":          ["tetracycline","tetraciclina"],
    "Oxitetraciclina":       ["oxytetracycline","oxitetraciclina"],
    "Clortetraciclina":      ["chlortetracycline","clortetraciclina"],
    "Doxiciclina":           ["doxycycline","doxiciclina"],
    "Ciprofloxacino":        ["ciprofloxacin","ciprofloxacino"],
    "Norfloxacino":          ["norfloxacin","norfloxacino"],
    "Ofloxacino":            ["ofloxacin","ofloxacino"],
    "Levofloxacino":         ["levofloxacin","levofloxacino"],
    "Sulfametoxazol":        ["sulfamethoxazole","sulfametoxazol"],
    "Trimetoprima":          ["trimethoprim","trimetoprima"],
    "Amoxicilina":           ["amoxicillin","amoxicilina"],
    "Ampicilina":            ["ampicillin","ampicilina"],
    "Cloranfenicol":         ["chloramphenicol","cloranfenicol"],
    "Eritromicina":          ["erythromycin","eritromicina"],
    "Azitromicina":          ["azithromycin","azitromicina"],
    "Estradiol (E2)":        ["estradiol","17beta-estradiol","17b-estradiol"],
    "Estrona (E1)":          ["estrone","estrona"],
    "Etinilestradiol (EE2)": ["ethinylestradiol","etinilestradiol"],
    "Estriol (E3)":          ["estriol"],
    "Progesterona":          ["progesterone","progesterona"],
    "Androgenos":            ["androgen","androgens","androgeno","testosterone","testosterona"],
}

def detect_analitos(texto):
    encontrados = []
    for nombre, kws in ANALITOS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(norm(kw)) + r"\b", texto):
                encontrados.append(nombre)
                break
    return "; ".join(encontrados) if encontrados else None

# Variables fisicoquímicas y estructurales
def extraer_phpzc(t):
    for patron in [r"ph[pz][zc][^0-9]{0,8}([\d]+\.?[\d]*)",
                   r"point of zero charge[^0-9]{0,8}([\d]+\.?[\d]*)",
                   r"pzc[^0-9]{0,8}([\d]+\.?[\d]*)"]:
        m = re.search(patron, t)
        if m: return float(m.group(1))
    return None

def extraer_bet(t):
    # Busca BET explícito o surface area con valor numérico razonable (1-3000 m2/g)
    for patron in [
        r"bet\s*(?:surface\s*area)?[^0-9]{0,10}([\d]+\.?[\d]*)\s*m[2²]",
        r"surface\s*area[^0-9]{0,10}([\d]+\.?[\d]*)\s*m[2²]\s*/\s*g",
        r"([\d]+\.?[\d]*)\s*m[2²]\s*/\s*g\s*(?:bet|surface)",
    ]:
        m = re.search(patron, t)
        if m:
            val = float(m.group(1))
            if 0.1 <= val <= 3000:  # rango físicamente razonable
                return val
    return None

def extraer_poro(t):
    m = re.search(r"pore\s*(?:size|diameter)[^0-9]{0,8}([\d]+\.?[\d]*)\s*(nm|a)", t)
    return float(m.group(1)) if m else None

def detect_grupos(t):
    grupos = []
    if re.search(r"\boh\b|hydroxyl", t):              grupos.append("OH")
    if re.search(r"\bcooh\b|carboxyl", t):            grupos.append("COOH")
    if re.search(r"\bnh[23]?\b|amine|amino", t):      grupos.append("NH")
    if re.search(r"\bc=o\b|carbonyl|ketone|aldehyde", t): grupos.append("C=O")
    if re.search(r"\bc=c\b|aromatic|benzene", t):     grupos.append("C=C")
    if re.search(r"\bso[23]?\b|sulfon", t):           grupos.append("SO3")
    return "; ".join(grupos) if grupos else None

def extraer_zeta(t):
    m = re.search(r"zeta[^0-9\-]{0,8}(-?[\d]+\.?[\d]*)\s*mv", t)
    return float(m.group(1)) if m else None

def extraer_ph_op(t):
    for patron in [
        r"optimum\s*ph[^0-9]{0,8}([\d]+\.?[\d]*)",
        r"optimal\s*ph[^0-9]{0,8}([\d]+\.?[\d]*)",
        r"ph\s*(?:of\s*)?(?:optimum|optimal)\s*(?:adsorption|removal)[^0-9]{0,8}([\d]+\.?[\d]*)",
        r"maximum\s*(?:removal|adsorption)\s*(?:at|was\s*achieved\s*at)\s*ph\s*([\d]+\.?[\d]*)",
        r"best\s*(?:removal|adsorption)\s*(?:at|was)\s*ph\s*([\d]+\.?[\d]*)",
    ]:
        m = re.search(patron, t)
        if m:
            val = float(m.group(1))
            if 1.0 <= val <= 14.0:  # rango físicamente válido
                return val
    return None

def extraer_dosis(t):
    m = re.search(r"([\d]+\.?[\d]*)\s*g\s*/\s*l", t)
    return float(m.group(1)) if m else None

def tipo_matriz(t):
    if re.search(r"real\s*wastewater|actual\s*wastewater|real\s*water|hospital\s*effluent|river\s*water\s*sample", t):
        return "Real"
    if re.search(r"synthetic|simulated|spiked|ultrapure|milli-?q|deionized", t):
        return "Sintetica"
    return None

def extraer_regen(t):
    m = re.search(r"([\d]+)\s*(?:regeneration|adsorption.desorption|cycle)", t)
    return int(m.group(1)) if m else None

def extraer_qmax(t):
    m = re.search(r"qmax[^0-9]{0,10}([\d]+\.?[\d]*)", t)
    return float(m.group(1)) if m else None

def validar_numero(valor):
    try:
        return float(valor)
    except ValueError:
        return None  # Si el valor no es un número, lo descartamos.

def extraer_qmax(t):
    m = re.search(r"qmax[^0-9]{0,10}([\d]+\.?[\d]*)", t)
    return validar_numero(m.group(1)) if m else None 
    
def extraer_remocion(t):
    m = re.search(r"([\d]{1,3}\.?[\d]*)\s*%\s*(removal|remocion|remocao|eliminacion)", t)
    return float(m.group(1)) if m else None

def detectar_isoterma(t):
    if "langmuir" in t and "freundlich" in t: return "Langmuir + Freundlich"
    if "langmuir"   in t: return "Langmuir"
    if "freundlich" in t: return "Freundlich"
    if "temkin"     in t: return "Temkin"
    if "sips"       in t: return "Sips"
    return None

def detectar_cinetica(t):
    pso = bool(re.search(r"pseudo.second.order|pso\b", t))
    pfo = bool(re.search(r"pseudo.first.order|pfo\b",  t))
    if pso and pfo: return "PFO + PSO"
    if pso: return "PSO"
    if pfo: return "PFO"
    return None

def tiene_cuant(t):
    return any(re.search(p, t) for p in [
        r"\bqmax\b", r"\bqe\b", r"\bmg\s*/\s*g\b", r"%\s*removal",
        r"removal efficiency", r"langmuir", r"freundlich",
        r"pseudo.second.order", r"pseudo.first.order",
    ])

# ── Aplicar todas las funciones a df_inc ─────────────────────────────────────
print("      Aplicando extraccion a articulos incluidos...")
df_inc["analitos"]       = df_inc["texto"].apply(detect_analitos)
df_inc["phpzc"]          = df_inc["texto"].apply(extraer_phpzc)
df_inc["bet_m2g"]        = df_inc["texto"].apply(extraer_bet)
df_inc["pore_nm"]        = df_inc["texto"].apply(extraer_poro)
df_inc["grupos_func"]    = df_inc["texto"].apply(detect_grupos)
df_inc["zeta_mv"]        = df_inc["texto"].apply(extraer_zeta)
df_inc["ph_operacional"] = df_inc["texto"].apply(extraer_ph_op)
df_inc["dosis_g_l"]      = df_inc["texto"].apply(extraer_dosis)
df_inc["tipo_matriz"]    = df_inc["texto"].apply(tipo_matriz)
df_inc["ciclos_regen"]   = df_inc["texto"].apply(extraer_regen)
df_inc["qmax_mg_g"]      = df_inc["texto"].apply(extraer_qmax)
df_inc["remocion_pct"]   = df_inc["texto"].apply(extraer_remocion)
df_inc["isoterma"]       = df_inc["texto"].apply(detectar_isoterma)
df_inc["cinetica"]       = df_inc["texto"].apply(detectar_cinetica)
df_inc["tiene_cuant"]    = df_inc["texto"].apply(tiene_cuant)

print(f"  OK  Variables extraidas en articulos incluidos:")
print(f"      Analito detectado      : {df_inc['analitos'].notna().sum():>6,}")
print(f"      pHpzc                  : {df_inc['phpzc'].notna().sum():>6,}")
print(f"      BET (m2/g)             : {df_inc['bet_m2g'].notna().sum():>6,}")
print(f"      Grupos funcionales     : {df_inc['grupos_func'].notna().sum():>6,}")
print(f"      pH operacional         : {df_inc['ph_operacional'].notna().sum():>6,}")
print(f"      Dosis (g/L)            : {df_inc['dosis_g_l'].notna().sum():>6,}")
print(f"      Tipo matriz real/sint  : {df_inc['tipo_matriz'].notna().sum():>6,}")
print(f"      Ciclos regeneracion    : {df_inc['ciclos_regen'].notna().sum():>6,}")
print(f"      qmax (mg/g)            : {df_inc['qmax_mg_g'].notna().sum():>6,}")
print(f"      % remocion             : {df_inc['remocion_pct'].notna().sum():>6,}")
print(f"      Isoterma               : {df_inc['isoterma'].notna().sum():>6,}")
print(f"      Cinetica               : {df_inc['cinetica'].notna().sum():>6,}")
print(f"      Con datos cuantitativos: {df_inc['tiene_cuant'].sum():>6,}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 9 — EXPORTAR LOS 10 CSVs
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{seg()} PASO 9/9 — Exportando CSVs...")

conteos = df_base["etiqueta"].value_counts().to_dict()

def guardar(df_, nombre, cols, descripcion=""):
    cols_ok = [c for c in cols if c in df_.columns]
    ruta = os.path.join(OUTPUT_DIR, nombre)
    df_[cols_ok].to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"  OK  {nombre:<45} {len(df_):>7,} filas   {descripcion}")

# 01 — Base completa
guardar(df_base, "01_todos_los_articulos.csv",
    ["title_raw","year","journal_raw","doi_raw","autor1",
     "etiqueta","score","razones","hits_obj","hits_agu","hits_ads",
     "exc_c","exc_e","exc_d","anio_ok","dup_doi","dup_fuzzy"],
    "(base completa)")

# 01B — Prioridad baja
df_baja = df_base[df_base["etiqueta"] == "PRIORIDAD_BAJA"].copy()
guardar(df_baja, "01b_prioridad_baja_ver.csv",
    ["title_raw","year","journal_raw","doi_raw","score","razones"],
    "(No descartar)")

# 02 — PRISMA
prisma = pd.DataFrame([
    {"etapa":"1. Registros cargados del RIS",                   "n": total_cargados},
    {"etapa":"2. Duplicados eliminados (DOI exacto)",           "n": n_dup_doi},
    {"etapa":"3. Duplicados eliminados (titulo similar)",       "n": n_dup_fuzzy},
    {"etapa":"4. Registros unicos analizados",                  "n": len(df_base)},
    {"etapa":"--- EXCLUIDOS ---",                               "n": ""},
    {"etapa":"5. Anio fuera de rango",                         "n": int((~df_base["anio_ok"]).sum())},
    {"etapa":"6. Tipo documental inadecuado",                  "n": int(df_base["exc_d"].sum())},
    {"etapa":"7. Contaminante fuera de foco",                  "n": int(df_base["exc_c"].sum())},
    {"etapa":"8. Tipo de estudio fuera de foco",               "n": int(df_base["exc_e"].sum())},
    {"etapa":"9. Total EXCLUIDOS",                             "n": conteos.get("EXCLUIDO",0)},
    {"etapa":"--- INCLUIDOS ---",                               "n": ""},
    {"etapa":"10. INCLUIDOS alta relevancia",                  "n": conteos.get("INCLUIDO_ALTA",0)},
    {"etapa":"11. INCLUIDOS media relevancia",                 "n": conteos.get("INCLUIDO_MEDIA",0)},
    {"etapa":"12. PRIORIDAD BAJA (revisar manualmente)",       "n": conteos.get("PRIORIDAD_BAJA",0)},
    {"etapa":"13. Con datos cuantitativos (meta-analisis)",    "n": int(df_inc["tiene_cuant"].sum())},
])
guardar(prisma, "02_prisma_flujo.csv", list(prisma.columns), "(diagrama PRISMA)")

# 03 — Por año
por_anio = df_inc.groupby("year").size().reset_index(name="n_articulos")
por_anio["pct"] = (por_anio["n_articulos"] / por_anio["n_articulos"].sum() * 100).round(1)
guardar(por_anio, "03_por_anio.csv", list(por_anio.columns), "(tendencia temporal)")

# 04 — Por país y continente
por_pais = (df_inc.groupby(["continente","pais"]).size()
            .reset_index(name="n_articulos")
            .sort_values("n_articulos", ascending=False))
por_pais["pct"] = (por_pais["n_articulos"] / len(df_inc) * 100).round(1)
guardar(por_pais, "04_por_pais_continente.csv", list(por_pais.columns), "(bibliometria geografica)")

# 05 — Por revista
por_revista = (df_inc.groupby("journal_raw").size()
               .reset_index(name="n_articulos")
               .sort_values("n_articulos", ascending=False))
por_revista["pct"] = (por_revista["n_articulos"] / len(df_inc) * 100).round(1)
guardar(por_revista, "05_por_revista.csv", list(por_revista.columns), "(clasificar base de datos en Excel)")

# 06 — Revision vs investigacion
guardar(df_inc, "06_revision_vs_investigacion.csv",
    ["title_raw","year","journal_raw","doi_raw","autor1","tipo_articulo","etiqueta","score"],
    "(revisiones = antecedentes)")

# 07 — Analitos + adsorbentes + autores
guardar(df_inc, "07_antibioticos_adsorbentes.csv",
    ["title_raw","year","journal_raw","doi_raw","autor1",
     "analitos","adsorbente","tipo_articulo","etiqueta","score",
     "db_raw"],          # ← AÑADIR
    "(analito + adsorbente + autor)")

# 08 — Variables fisicoquímicas completas
guardar(df_inc, "08_variables_fisicoquimicas.csv",
    ["title_raw","year","doi_raw","autor1","analitos","adsorbente",
     "phpzc","bet_m2g","pore_nm","grupos_func","zeta_mv",
     "ph_operacional","dosis_g_l","tipo_matriz","ciclos_regen",
     "qmax_mg_g","remocion_pct","isoterma","cinetica"],
    "(variables fisicoquimicas y operacionales)")

# 09 — Solo con datos cuantitativos
df_cuant = df_inc[df_inc["tiene_cuant"] == True].copy()
guardar(df_cuant, "09_datos_cuantitativos.csv",
    ["title_raw","year","journal_raw","doi_raw","pais","adsorbente","analitos",
     "qmax_mg_g","remocion_pct","isoterma","cinetica",
     "phpzc","bet_m2g","ph_operacional","dosis_g_l","tipo_matriz","ciclos_regen",
     "etiqueta","score"],
    "(para metafor en R)")

# 10 — Bibliometrico general
guardar(df_inc, "10_bibliometrico.csv",
    ["title_raw","year","journal_raw","doi_raw",
     "autor1","authors_raw",    # ← añadir authors_raw
     "pais","continente","adsorbente","analitos",
     "tipo_articulo","etiqueta","score","db_raw"],
    "(para bibliometrix en R, autores)")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
t_total = time.time() - t0
titulo_seccion("RESUMEN FINAL")
print(f"  Tiempo total                  : {t_total:.0f} segundos ({t_total/60:.1f} min)")
print(f"  Registros cargados (RIS)      : {total_cargados:>7,}")
print(f"  Duplicados eliminados (DOI)   : {n_dup_doi:>7,}")
print(f"  Duplicados eliminados (fuzzy) : {n_dup_fuzzy:>7,}")
print(f"  Registros unicos analizados   : {len(df_base):>7,}")
print(f"  ─────────────────────────────────────────────")
print(f"  EXCLUIDOS                     : {conteos.get('EXCLUIDO',0):>7,}")
print(f"  INCLUIDOS alta relevancia     : {conteos.get('INCLUIDO_ALTA',0):>7,}")
print(f"  INCLUIDOS media relevancia    : {conteos.get('INCLUIDO_MEDIA',0):>7,}")
print(f"  PRIORIDAD BAJA (revisar)      : {conteos.get('PRIORIDAD_BAJA',0):>7,}")
print(f"  ─────────────────────────────────────────────")
print(f"  Total incluidos               : {len(df_inc):>7,}")
print(f"    Revisiones (antecedentes)   : {(df_inc['tipo_articulo']=='Revision').sum():>7,}")
print(f"    Investigacion original      : {(df_inc['tipo_articulo']=='Investigacion original').sum():>7,}")
print(f"  ─────────────────────────────────────────────")
print(f"  Con analito detectado         : {df_inc['analitos'].notna().sum():>7,}")
print(f"  Con pHpzc                     : {df_inc['phpzc'].notna().sum():>7,}")
print(f"  Con BET (m2/g)                : {df_inc['bet_m2g'].notna().sum():>7,}")
print(f"  Con grupos funcionales        : {df_inc['grupos_func'].notna().sum():>7,}")
print(f"  Con pH operacional            : {df_inc['ph_operacional'].notna().sum():>7,}")
print(f"  Con tipo de matriz            : {df_inc['tipo_matriz'].notna().sum():>7,}")
print(f"  Con ciclos regeneracion       : {df_inc['ciclos_regen'].notna().sum():>7,}")
print(f"  Con qmax (mg/g)               : {df_inc['qmax_mg_g'].notna().sum():>7,}")
print(f"  Con % remocion                : {df_inc['remocion_pct'].notna().sum():>7,}")
print(f"  Con datos cuantitativos total : {df_inc['tiene_cuant'].sum():>7,}")
print(f"  ─────────────────────────────────────────────")
print(f"  Top 5 paises:")
for p, n in df_inc["pais"].value_counts().head(5).items():
    cont = CONTINENTE.get(str(p), "?")
    print(f"    {str(p):<30} {n:>5}  ({cont})")
print(f"  Top 5 continentes:")
for c, n in df_inc["continente"].value_counts().head(5).items():
    print(f"    {str(c):<30} {n:>5}")
print(f"  Top 5 adsorbentes:")
for a, n in df_inc["adsorbente"].value_counts().head(5).items():
    print(f"    {str(a):<30} {n:>5}")
print(f"  Top 5 analitos:")
analitos_exp = df_inc["analitos"].dropna().str.split("; ").explode()
for a, n in analitos_exp.value_counts().head(5).items():
    print(f"    {str(a):<30} {n:>5}")
print(f"  Top 5 revistas:")
for r, n in df_inc["journal_raw"].value_counts().head(5).items():
    print(f"    {str(r)[:52]:<52} {n:>5}")
titulo_seccion(f"10 archivos guardados en ./{OUTPUT_DIR}/")
