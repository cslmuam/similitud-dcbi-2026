#!/usr/bin/env python3
"""regenerar_figuras.py — Rediseño de 14 figuras del reporte institucional DCBI 2026.

Genera en Figuras/ con los mismos nombres que ya existen (para no romper el LaTeX):
  1.  heatmap_licenciaturas.png
  2.  heatmap_interplan_conteo.png
  3.  heatmap_clustered.png
  4.  scatter_tsne.png
  5-14. intraplan_<Licenciatura>.png  ×10

NO regenera el grafo de licenciaturas (variante_B ya está aprobado).
"""

from __future__ import annotations

import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REVISOR = Path("/Users/nowhere/Claude_Projects/Admin_duties/Modificaciones2026/revisor")
REVISIONES = Path("/Users/nowhere/Claude_Projects/Admin_duties/Modificaciones2026/Revisiones")
SALIDA = REVISIONES / "Figuras"
SALIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REVISOR))

# ── Paleta canónica ───────────────────────────────────────────────────────────
COLORES: dict[str, str] = {
    "Ambiental":   "#2E7D32",
    "Civil":       "#BF360C",
    "Computación": "#1565C0",
    "Eléctrica":   "#F9A825",
    "Electrónica": "#6A1B9A",
    "Física":      "#00838F",
    "Industrial":  "#C62828",
    "Mecánica":    "#37474F",
    "Metalúrgica": "#78909C",
    "Química":     "#558B2F",
}

ORDEN = [
    "Ambiental", "Civil", "Computación", "Eléctrica", "Electrónica",
    "Física", "Industrial", "Mecánica", "Metalúrgica", "Química",
]


def _c(lic: str) -> str:
    return COLORES.get(lic, "#999999")


# ── Estilo global ─────────────────────────────────────────────────────────────
def _setup_rcparams() -> None:
    plt.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor":      "#FFFFFF",
        "axes.facecolor":        "#FFFFFF",
        "savefig.facecolor":     "#FFFFFF",
        "axes.edgecolor":        "#CCCCCC",
        "axes.linewidth":        0.8,
        "xtick.color":           "#444444",
        "ytick.color":           "#444444",
        "text.color":            "#222222",
    })


# ── Utilidad: color de texto sobre celda ──────────────────────────────────────
def _text_color(hex_bg: str | None = None, val: float | None = None,
                vmin: float = 0.0, vmax: float = 1.0,
                cmap=None) -> str:
    """Devuelve 'black' o 'white' según la luminancia del fondo."""
    if hex_bg is not None:
        r = int(hex_bg[1:3], 16) / 255
        g = int(hex_bg[3:5], 16) / 255
        b = int(hex_bg[5:7], 16) / 255
    elif cmap is not None and val is not None:
        t = (val - vmin) / max(vmax - vmin, 1e-9)
        rgba = cmap(t)
        r, g, b = rgba[0], rgba[1], rgba[2]
    else:
        return "black"
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= 0.40 else "white"


# ══════════════════════════════════════════════════════════════════════════════
# Carga de datos
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos():
    """Carga corpus + MatrizHibrida desde los archivos persistidos."""
    from config import LICENCIATURAS
    from agentes.inventario import inventariar_todos
    from agentes.extractor_contenido import extraer_corpus
    from agentes.similitud_llm import MatrizHibrida

    print("  Inventariando documentos...", flush=True)
    inventarios = inventariar_todos(LICENCIATURAS)

    print("  Extrayendo corpus (excluyendo TG)...", flush=True)
    corpus, _, _ = extraer_corpus(inventarios, umbral_tg=7)
    print(f"  Corpus: {len(corpus)} UEAs", flush=True)

    base = REVISIONES / "Datos" / "similitud_llm_dcbi_2026"
    print("  Cargando MatrizHibrida...", flush=True)
    mh = MatrizHibrida.cargar(corpus, base)
    print(f"  Matriz: {mh.matriz.shape}  ({mh.n_evaluados} pares LLM)", flush=True)

    return corpus, mh


# ══════════════════════════════════════════════════════════════════════════════
# 1. heatmap_licenciaturas.png
# ══════════════════════════════════════════════════════════════════════════════

def fig1_heatmap_licenciaturas(corpus, mat: np.ndarray) -> Path:
    _setup_rcparams()

    idx_por_lic: dict[str, list[int]] = defaultdict(list)
    for i, cu in enumerate(corpus):
        idx_por_lic[cu.licenciatura].append(i)

    lics = [l for l in ORDEN if idx_por_lic.get(l)]
    n = len(lics)

    agg = np.zeros((n, n), dtype=np.float64)
    for i, la in enumerate(lics):
        for j, lb in enumerate(lics):
            if i == j:
                agg[i, j] = 1.0
                continue
            ia = idx_por_lic[la]
            ib = idx_por_lic[lb]
            if not ia or not ib:
                continue
            sub = mat[np.ix_(ia, ib)]
            agg[i, j] = float(sub.max(axis=1).mean())

    # Encontrar celda fuera de diagonal con valor máximo
    mask_off = ~np.eye(n, dtype=bool)
    max_off_val = agg[mask_off].max()
    max_off_pos = [(i, j) for i in range(n) for j in range(n)
                   if i != j and abs(agg[i, j] - max_off_val) < 1e-6]

    cmap = plt.cm.Reds
    fig, ax = plt.subplots(figsize=(10, 9))

    # Celdas fuera de la diagonal con colormap
    data_plot = np.ma.array(agg, mask=np.eye(n, dtype=bool))
    im = ax.imshow(data_plot, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Diagonal: color neutro oscuro
    for k in range(n):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                                color="#37474F", zorder=2))
        ax.text(k, k, "—", ha="center", va="center",
                fontsize=9, color="#ECEFF1", zorder=3)

    # Anotaciones off-diagonal
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = agg[i, j]
            tc = _text_color(val=v, vmin=0, vmax=1, cmap=cmap)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=tc, zorder=3)

    # Resaltar la celda inter-plan más alta con borde rojo
    for (ri, ci) in max_off_pos:
        for dr, dc in [(0, 0)]:
            ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                    fill=False, edgecolor="#D32F2F",
                                    linewidth=2.5, zorder=4))

    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("Similitud media-máxima", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(lics, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(lics, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Similitud de contenidos entre licenciaturas DCBI 2026\n"
        "Para cada UEA de la fila: máxima similitud con alguna UEA de la columna, promediada",
        fontsize=11, pad=14,
    )

    ruta = SALIDA / "heatmap_licenciaturas.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 2. heatmap_interplan_conteo.png
# ══════════════════════════════════════════════════════════════════════════════

def fig2_heatmap_interplan_conteo(corpus, mat: np.ndarray) -> Path:
    _setup_rcparams()

    idx_por_lic: dict[str, list[int]] = defaultdict(list)
    for i, cu in enumerate(corpus):
        idx_por_lic[cu.licenciatura].append(i)

    lics = [l for l in ORDEN if idx_por_lic.get(l)]
    n = len(lics)

    # Construir conteo inter-plan para dos umbrales
    def conteo_inter(umbral: float) -> np.ndarray:
        cnt = np.zeros((n, n), dtype=int)
        for ii, la in enumerate(lics):
            for jj, lb in enumerate(lics):
                if ii == jj:
                    continue
                ia = idx_por_lic[la]
                ib = idx_por_lic[lb]
                if not ia or not ib:
                    continue
                sub = mat[np.ix_(ia, ib)]
                cnt[ii, jj] = int((sub >= umbral).sum())
        return cnt

    cnt_85 = conteo_inter(0.85)
    cnt_70 = conteo_inter(0.70)

    vmax_global = max(int(cnt_85.max()), int(cnt_70.max()), 1)
    # Encontrar celda máxima en ambos paneles para resaltar
    max_85 = cnt_85[~np.eye(n, dtype=bool)].max()
    max_70 = cnt_70[~np.eye(n, dtype=bool)].max()

    cmap = plt.cm.Reds

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.12, right=0.88)

    def _dibujar_panel(ax, cnt: np.ndarray, titulo: str, umbral_val: float):
        data_plot = np.ma.array(cnt.astype(float), mask=np.eye(n, dtype=bool))
        im = ax.imshow(data_plot, cmap=cmap, vmin=0, vmax=vmax_global,
                       aspect="auto")

        # Diagonal gris
        for k in range(n):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                                    color="#BDBDBD", zorder=2))
            ax.text(k, k, "—", ha="center", va="center",
                    fontsize=8, color="#555555", zorder=3)

        # Anotaciones
        max_off = cnt[~np.eye(n, dtype=bool)].max()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = cnt[i, j]
                tc = _text_color(val=float(v), vmin=0, vmax=vmax_global, cmap=cmap)
                fw = "bold" if v >= 10 else "normal"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color=tc, fontweight=fw, zorder=3)
                # Resaltar celda máxima con borde rojo
                if v == max_off and v > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#D32F2F",
                                            linewidth=2.0,
                                            linestyle="-", zorder=4))

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(lics, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(lics, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(titulo, fontsize=10, pad=10)
        return im

    im1 = _dibujar_panel(axes[0], cnt_85,
                          "Pares inter-plan con $s \\geq 0.85$", 0.85)
    im2 = _dibujar_panel(axes[1], cnt_70,
                          "Pares inter-plan con $s \\geq 0.70$", 0.70)

    # Colorbar compartida
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("Número de pares", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Conteo de pares similares inter-plan — DCBI 2026",
                 fontsize=12, y=1.01)

    ruta = SALIDA / "heatmap_interplan_conteo.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 3. heatmap_clustered.png
# ══════════════════════════════════════════════════════════════════════════════

def fig3_heatmap_clustered(corpus, mat: np.ndarray,
                            umbral_cluster: float = 0.35,
                            max_ueas: int = 120) -> Path:
    _setup_rcparams()
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # Seleccionar las más conectadas
    n_full = len(corpus)
    grado: dict[int, int] = defaultdict(int)
    for i in range(n_full):
        for j in range(i + 1, n_full):
            if mat[i, j] >= umbral_cluster:
                grado[i] += 1
                grado[j] += 1

    idxs_sorted = sorted(grado, key=lambda i: -grado[i])[:max_ueas]
    idxs = sorted(idxs_sorted)
    n = len(idxs)

    sub = mat[np.ix_(idxs, idxs)].copy().astype(np.float64)

    # Clustering Ward
    dist = np.clip(1.0 - sub, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    sub_ord = sub[np.ix_(order, order)]
    np.fill_diagonal(sub_ord, 1.0)
    corp_ord  = [corpus[idxs[i]] for i in order]
    lics_ord  = [cu.licenciatura for cu in corp_ord]
    claves_ord = [cu.clave for cu in corp_ord]

    # ── Figura con barra lateral de licenciatura ──────────────────────────────
    lic_rgb = np.array([
        [int(_c(l).lstrip("#")[k:k + 2], 16) / 255 for k in (0, 2, 4)]
        for l in lics_ord
    ])  # (n, 3)

    cell = max(0.10, min(0.18, 14.0 / n))
    fig_w = max(14, n * cell + 2.5)
    fig_h = max(12, n * cell + 1.5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    # width_ratios: barra lateral | heatmap | colorbar
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.042, 1, 0.025],
        height_ratios=[1, 0.045],
        hspace=0.02, wspace=0.025,
    )
    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_cb   = fig.add_subplot(gs[0, 2])
    ax_leg  = fig.add_subplot(gs[1, :])

    cmap = plt.cm.Reds
    im = ax_heat.imshow(sub_ord, cmap=cmap, vmin=0, vmax=1,
                        aspect="auto", interpolation="nearest")
    plt.colorbar(im, cax=ax_cb, label="Similitud coseno")
    ax_cb.tick_params(labelsize=8)
    ax_cb.yaxis.label.set_size(9)

    if n <= 80:
        ax_heat.set_xticks(range(n))
        ax_heat.set_yticks(range(n))
        ax_heat.set_xticklabels(claves_ord, rotation=90, fontsize=5,
                                 fontfamily="monospace")
        ax_heat.set_yticklabels(claves_ord, fontsize=5,
                                 fontfamily="monospace")
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    ax_heat.set_title(
        f"Heatmap clustered — {n} UEAs con mayor traslape (sim ≥ {umbral_cluster})\n"
        "Reordenamiento jerárquico de Ward · color por licenciatura en barra lateral",
        fontsize=10, pad=10,
    )

    # Barra lateral con la paleta canónica
    ax_bar.imshow(lic_rgb.reshape(n, 1, 3), aspect="auto", interpolation="nearest")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Leyenda inferior con paleta canónica
    lics_en = sorted({l for l in lics_ord})
    patches = [
        mpatches.Patch(color=_c(l), label=l,
                       linewidth=0.5, edgecolor="#FFFFFF")
        for l in ORDEN if l in set(lics_en)
    ]
    ax_leg.axis("off")
    ax_leg.legend(
        handles=patches,
        loc="center",
        ncol=min(len(patches), 10),
        fontsize=8.5,
        framealpha=0,
        handlelength=1.2,
        handleheight=1.2,
    )

    ruta = SALIDA / "heatmap_clustered.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 4. scatter_tsne.png
# ══════════════════════════════════════════════════════════════════════════════

def fig4_scatter_tsne(corpus, mat: np.ndarray,
                      umbral_lineas: float = 0.70) -> Path:
    _setup_rcparams()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    _STOPWORDS = {
        "a","al","algo","algunas","algunos","ante","antes","como","con","contra",
        "cual","cuando","de","del","desde","donde","durante","e","el","ella",
        "ellas","ellos","en","entre","era","es","esa","ese","esta","este","fue",
        "ha","han","has","hasta","hay","la","las","le","les","lo","los","me",
        "mi","mis","muy","más","ni","no","nos","o","para","pero","por","que",
        "se","sea","ser","si","sin","sobre","son","su","sus","también","te",
        "ti","todo","todos","tu","tus","un","una","uno","unos","y","ya","yo",
        "objetivo","objetivos","general","parcial","contenido","temario",
        "enseñanza","aprendizaje","evaluación","evaluacion","modalidad",
        "estudiante","estudiantes","alumno","alumnos","curso","unidad",
        "práctica","practicas","teoría","teoria","horas","créditos","creditos",
        "programa","programas","ingeniería","ingenieria","introducción",
        "fundamentos","aplicaciones","conceptos","básicos","nivel","licenciatura",
    }

    textos = [cu.texto_completo for cu in corpus]
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=8000,
        sublinear_tf=True, stop_words=list(_STOPWORDS),
        min_df=2, max_df=0.85,
        token_pattern=r"(?u)\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{3,}\b",
    )
    X = vec.fit_transform(textos)

    svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42)
    X50 = svd.fit_transform(X)
    coords = TSNE(n_components=2, perplexity=30, random_state=42,
                  max_iter=1000, learning_rate="auto", init="pca").fit_transform(X50)

    xs, ys = coords[:, 0], coords[:, 1]
    lics = [cu.licenciatura for cu in corpus]
    n = len(corpus)

    # Pares inter-plan con s >= umbral_lineas
    pares_altos = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (float(mat[i, j]) >= umbral_lineas
            and corpus[i].licenciatura != corpus[j].licenciatura)
    ]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

    # Líneas de aristas inter-plan (fondo)
    for i, j in pares_altos:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color="#B0BEC5", linewidth=0.6, alpha=0.25, zorder=1)

    # Puntos por licenciatura
    lics_presentes = [l for l in ORDEN if l in set(lics)]
    for lic in lics_presentes:
        mask = np.array([l == lic for l in lics])
        ax.scatter(xs[mask], ys[mask],
                   c=_c(lic), label=lic, s=45, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=2)

    # Ejes limpios: sin ticks, sin tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_xlabel("t-SNE 1", fontsize=9, color="#666666")
    ax.set_ylabel("t-SNE 2", fontsize=9, color="#666666")

    # Leyenda 2 columnas
    leg = ax.legend(
        loc="upper left", fontsize=9,
        markerscale=0.6, framealpha=0.0,
        ncol=2, columnspacing=0.8, handletextpad=0.4,
        borderpad=0.4,
    )
    for h in leg.legend_handles:
        try:
            h.set_sizes([64])   # PathCollection: tamaño en puntos²
        except AttributeError:
            try:
                h.set_markersize(8)
            except AttributeError:
                pass

    ax.set_title(
        "Proyección t-SNE — Corpus DCBI 2026\n"
        r"Un punto por UEA · color por licenciatura · "
        f"líneas = pares inter-plan con $s \\geq {umbral_lineas}$",
        fontsize=10, pad=12,
    )

    ruta = SALIDA / "scatter_tsne.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 5–14. intraplan_*.png
# ══════════════════════════════════════════════════════════════════════════════

_SLUG = {
    "Ambiental":   "Ambiental",
    "Civil":       "Civil",
    "Computación": "Computacion",
    "Eléctrica":   "Electrica",
    "Electrónica": "Electronica",
    "Física":      "Fisica",
    "Industrial":  "Industrial",
    "Mecánica":    "Mecanica",
    "Metalúrgica": "Metalurgica",
    "Química":     "Quimica",
}


def fig_intraplan(corpus, mat: np.ndarray, lic: str,
                  umbral_resalte: float = 0.85) -> Path:
    _setup_rcparams()
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    idxs = [i for i, cu in enumerate(corpus) if cu.licenciatura == lic]
    n = len(idxs)
    if n < 3:
        raise ValueError(f"Solo {n} UEAs para {lic}")

    sub = mat[np.ix_(idxs, idxs)].copy().astype(np.float64)
    np.fill_diagonal(sub, 1.0)

    claves = [corpus[i].clave for i in idxs]

    # Clustering Ward
    dist = np.clip(1.0 - sub, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    sub_ord = sub[np.ix_(order, order)]
    np.fill_diagonal(sub_ord, 1.0)
    claves_ord = [claves[i] for i in order]

    # Contar pares s >= 0.40 (fuera de diagonal)
    mask_off = ~np.eye(n, dtype=bool)
    n_pares_040 = int(((sub_ord[mask_off]) >= 0.40).sum()) // 2

    # Tamaño adaptativo
    sz = max(7.0, min(12.0, n / 10.0))
    fig, ax = plt.subplots(figsize=(sz + 1.0, sz), facecolor="white")

    cmap = plt.cm.Reds

    # Dibujar heatmap (sin diagonal primero)
    data_plot = np.ma.array(sub_ord, mask=np.eye(n, dtype=bool))
    im = ax.imshow(data_plot, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    # Diagonal: negro/gris oscuro
    for k in range(n):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                                color="#212121", zorder=2))

    # Resaltar pares s >= umbral_resalte (fuera diagonal) con rectángulo rojo punteado
    color_lic = _c(lic)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if sub_ord[i, j] >= umbral_resalte:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#D32F2F",
                    linewidth=1.2, linestyle="--", zorder=4,
                ))

    # Etiquetas de UEA si n <= 45
    if n <= 45:
        fs = max(5, min(7, int(42 / n)))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(claves_ord, rotation=90, fontsize=fs,
                           fontfamily="monospace")
        ax.set_yticklabels(claves_ord, fontsize=fs,
                           fontfamily="monospace")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Similitud coseno", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Título con recuento
    ax.set_title(
        f"{lic} — similitud intra-plan\n"
        f"{n} UEAs · {n_pares_040} pares con $s \\geq 0.40$",
        fontsize=11, pad=12, color=color_lic,
    )

    slug = _SLUG.get(lic, lic)
    ruta = SALIDA / f"intraplan_{slug}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 65)
    print("  Regeneración de figuras — DCBI 2026")
    print("=" * 65 + "\n")

    corpus, mh = cargar_datos()
    mat = mh.matriz

    resultados: list[tuple[str, bool, str]] = []

    # ── 1. Heatmap licenciaturas 10×10 ────────────────────────────────────────
    print("[1/14] heatmap_licenciaturas.png ...", flush=True)
    try:
        p = fig1_heatmap_licenciaturas(corpus, mat)
        print(f"       ✓ {p.name}")
        resultados.append(("heatmap_licenciaturas.png", True, ""))
    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        resultados.append(("heatmap_licenciaturas.png", False, str(e)))

    # ── 2. Heatmap interplan conteo ───────────────────────────────────────────
    print("[2/14] heatmap_interplan_conteo.png ...", flush=True)
    try:
        p = fig2_heatmap_interplan_conteo(corpus, mat)
        print(f"       ✓ {p.name}")
        resultados.append(("heatmap_interplan_conteo.png", True, ""))
    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        resultados.append(("heatmap_interplan_conteo.png", False, str(e)))

    # ── 3. Heatmap clustered ──────────────────────────────────────────────────
    print("[3/14] heatmap_clustered.png ...", flush=True)
    try:
        p = fig3_heatmap_clustered(corpus, mat)
        print(f"       ✓ {p.name}")
        resultados.append(("heatmap_clustered.png", True, ""))
    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        resultados.append(("heatmap_clustered.png", False, str(e)))

    # ── 4. Scatter t-SNE ──────────────────────────────────────────────────────
    print("[4/14] scatter_tsne.png  (t-SNE, puede tardar ~60s) ...", flush=True)
    try:
        p = fig4_scatter_tsne(corpus, mat)
        print(f"       ✓ {p.name}")
        resultados.append(("scatter_tsne.png", True, ""))
    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        resultados.append(("scatter_tsne.png", False, str(e)))

    # ── 5–14. Heatmaps intra-plan ─────────────────────────────────────────────
    for idx, lic in enumerate(ORDEN, start=5):
        nombre_fig = f"intraplan_{_SLUG[lic]}.png"
        print(f"[{idx}/14] {nombre_fig} ...", flush=True)
        try:
            p = fig_intraplan(corpus, mat, lic)
            print(f"       ✓ {p.name}")
            resultados.append((nombre_fig, True, ""))
        except Exception as e:
            print(f"       ✗ ERROR: {e}")
            resultados.append((nombre_fig, False, str(e)))

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    ok = [r for r in resultados if r[1]]
    fail = [r for r in resultados if not r[1]]
    print(f"  Exitosas : {len(ok)}/14")
    if fail:
        print(f"  Fallidas : {len(fail)}/14")
        for nombre, _, err in fail:
            print(f"    ✗ {nombre}: {err}")
    else:
        print("  Sin errores.")
    print(f"\n  Figuras en: {SALIDA}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
