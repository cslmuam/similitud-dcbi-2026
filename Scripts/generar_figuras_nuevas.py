#!/usr/bin/env python3
"""generar_figuras_nuevas.py — Genera cuatro figuras nuevas para el dashboard DCBI 2026.

Figuras generadas (PNG + HTML interactivo):
  A1. barras_interplan_top.png / .html — Pares equivalentes inter-plan (s >= 0.70)
  A2. scatter_tfidf_llm.png / .html     — Scatter score híbrido vs. score LLM
  A3. matriz_transversal.png / .html    — Heatmap UEAs presentes en 5+ licenciaturas
  A4. histograma_intraplan.png / .html  — Distribución KDE intra-plan por licenciatura

Inputs:
  Revisiones/Datos/similitud_llm_dcbi_2026.npz  — matrices 626×626 (híbrida + tfidf)
  Revisiones/Datos/similitud_llm_dcbi_2026.json — 1505 pares evaluados por LLM
  revisor/ — módulos corpus (corpus se carga via extraer_corpus)

Outputs:
  Revisiones/Figuras/*.png  (DPI 200)
  Revisiones/Figuras/*.html (Plotly standalone)
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/Users/nowhere/Claude_Projects/Admin_duties/Modificaciones2026")
REVISOR   = BASE / "revisor"
REVISIONES = BASE / "Revisiones"
DATOS     = REVISIONES / "Datos"
SALIDA    = REVISIONES / "Figuras"
SALIDA.mkdir(parents=True, exist_ok=True)

NPZ_FILE  = DATOS / "similitud_llm_dcbi_2026.npz"
JSON_FILE = DATOS / "similitud_llm_dcbi_2026.json"

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

ACCENT = "#C0101A"


def _c(lic: str) -> str:
    return COLORES.get(lic, "#999999")


# ── Estilo global ─────────────────────────────────────────────────────────────
def _setup_rcparams() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor":  "#FFFFFF",
        "axes.facecolor":    "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor":    "#CCCCCC",
        "axes.linewidth":    0.8,
        "xtick.color":       "#444444",
        "ytick.color":       "#444444",
        "text.color":        "#0D0D0D",
    })


# ── Escritura atómica ─────────────────────────────────────────────────────────
def _guardar_fig(fig: plt.Figure, ruta: Path, dpi: int = 200) -> Path:
    """Guarda figura PNG de forma atómica (temp → rename)."""
    tmp = ruta.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    tmp.rename(ruta)
    assert ruta.exists() and ruta.stat().st_size > 0, f"Archivo vacío: {ruta}"
    return ruta


def _guardar_html(html_str: str, ruta: Path) -> Path:
    """Guarda HTML Plotly standalone de forma atómica."""
    tmp = ruta.with_suffix(".tmp.html")
    tmp.write_text(html_str, encoding="utf-8")
    tmp.rename(ruta)
    assert ruta.exists() and ruta.stat().st_size > 0, f"Archivo vacío: {ruta}"
    return ruta


# ── Carga de datos ────────────────────────────────────────────────────────────
def cargar_datos():
    """Devuelve corpus, mat_hibrida, mat_tfidf, pares_llm."""
    log.info("Cargando corpus desde revisor/...")
    from config import LICENCIATURAS
    from agentes.inventario import inventariar_todos
    from agentes.extractor_contenido import extraer_corpus

    inventarios = inventariar_todos(LICENCIATURAS)
    corpus, _, _ = extraer_corpus(inventarios, umbral_tg=7)
    log.info("  Corpus: %d UEAs", len(corpus))

    log.info("Cargando matrices NPZ...")
    npz = np.load(NPZ_FILE)
    mat_hibrida = npz["matriz"].astype(np.float64)
    mat_tfidf   = npz["matriz_tfidf"].astype(np.float64)
    log.info("  Matrices: %s", mat_hibrida.shape)

    log.info("Cargando JSON LLM...")
    with open(JSON_FILE, encoding="utf-8") as f:
        jdata = json.load(f)
    pares_llm = jdata["puntuaciones"]
    log.info("  Pares LLM: %d", len(pares_llm))

    return corpus, mat_hibrida, mat_tfidf, pares_llm


# ══════════════════════════════════════════════════════════════════════════════
# A1. Bar chart horizontal — pares inter-plan >= 0.70
# ══════════════════════════════════════════════════════════════════════════════

def fig_A1_barras_interplan(corpus, mat: np.ndarray) -> tuple[Path, Path]:
    """Bar chart horizontal: conteo de pares equivalentes por par de licenciaturas."""
    import plotly.graph_objects as go

    _setup_rcparams()
    n = len(corpus)

    # Contar pares inter-plan con score >= 0.70
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] >= 0.70 and corpus[i].licenciatura != corpus[j].licenciatura:
                la, lb = sorted([corpus[i].licenciatura, corpus[j].licenciatura])
                pair_counts[(la, lb)] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1])
    labels = [f"{la} × {lb}" for (la, lb), _ in sorted_pairs]
    counts = [cnt for _, cnt in sorted_pairs]
    # Color = licenciatura con más UEAs del par (proxy: ORDEN index menor = la)
    bar_colors = []
    for (la, lb), _ in sorted_pairs:
        idx_a = ORDEN.index(la) if la in ORDEN else 99
        idx_b = ORDEN.index(lb) if lb in ORDEN else 99
        bar_colors.append(_c(la if idx_a <= idx_b else lb))

    # ── PNG (matplotlib) ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.35)), facecolor="white")
    bars = ax.barh(labels, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    # Anotaciones
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left", fontsize=8, color="#0D0D0D",
        )
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_xlabel("Número de pares equivalentes", fontsize=10)
    ax.set_title(
        "Pares equivalentes entre licenciaturas (s ≥ 0.70)",
        fontsize=12, pad=12,
    )
    ax.axvline(x=0, color="#CCCCCC", linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ruta_png = _guardar_fig(fig, SALIDA / "barras_interplan_top.png")
    plt.close(fig)
    log.info("  PNG: %s", ruta_png)

    # ── HTML (Plotly) ─────────────────────────────────────────────────────────
    # Reversed order for Plotly (top to bottom = highest first)
    fig_pl = go.Figure(go.Bar(
        y=labels[::-1],
        x=counts[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[str(c) for c in counts[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Pares: %{x}<extra></extra>",
    ))
    fig_pl.update_layout(
        title=dict(
            text="Pares equivalentes entre licenciaturas (s ≥ 0.70)",
            font=dict(size=14),
        ),
        xaxis_title="Número de pares equivalentes",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0D0D0D", size=11),
        margin=dict(l=200, r=80, t=60, b=50),
        height=max(400, len(labels) * 22 + 120),
        showlegend=False,
    )
    fig_pl.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    fig_pl.update_yaxes(showgrid=False)

    ruta_html = _guardar_html(fig_pl.to_html(full_html=True, include_plotlyjs="cdn"), SALIDA / "barras_interplan_top.html")
    log.info("  HTML: %s", ruta_html)

    return ruta_png, ruta_html


# ══════════════════════════════════════════════════════════════════════════════
# A2. Scatter score híbrido vs. score LLM
# ══════════════════════════════════════════════════════════════════════════════

def fig_A2_scatter_tfidf_llm(corpus, mat_hibrida: np.ndarray, mat_tfidf: np.ndarray, pares_llm: dict) -> tuple[Path, Path]:
    """Scatter híbrido vs. LLM para los 1505 pares candidatos."""
    import plotly.graph_objects as go

    _setup_rcparams()

    # Construir arrays desde los pares LLM evaluados
    hibridos, tfidf_vals, llm_vals, niveles_list = [], [], [], []
    clave_a_list, clave_b_list = [], []

    for key, v in pares_llm.items():
        try:
            i, j = [int(x) for x in key.split(",")]
        except ValueError:
            continue
        if i >= len(corpus) or j >= len(corpus):
            continue
        hibridos.append(float(mat_hibrida[i, j]))
        tfidf_vals.append(float(mat_tfidf[i, j]))
        llm_vals.append(float(v["similitud"]))
        niveles_list.append(v["nivel"])
        clave_a_list.append(corpus[i].clave)
        clave_b_list.append(corpus[j].clave)

    hibridos  = np.array(hibridos)
    tfidf_vals = np.array(tfidf_vals)
    llm_vals   = np.array(llm_vals)

    # Nivel → color (escala gris + acento para idéntico)
    nivel_colores = {
        "sin_traslape": "#BDBDBD",
        "bajo":         "#9E9E9E",
        "moderado":     "#616161",
        "alto":         "#424242",
        "muy_alto":     "#212121",
        "idéntico":     ACCENT,
    }
    nivel_order = ["sin_traslape", "bajo", "moderado", "alto", "muy_alto", "idéntico"]
    punto_colors = [nivel_colores.get(nv, "#999999") for nv in niveles_list]

    # ── PNG ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")

    # Sombrear región TF-IDF < 0.40 y LLM >= 0.40 (pares detectados por semántica)
    ax.axvspan(0, 0.40, alpha=0.04, color="#1565C0", zorder=0)
    ax.axhspan(0.40, 1.0, alpha=0.04, color="#1565C0", zorder=0)
    ax.axvline(0.40, color="#90CAF9", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.40, color="#90CAF9", linewidth=0.8, linestyle="--", zorder=1)

    # Línea y=x de referencia
    ax.plot([0, 1], [0, 1], color="#CCCCCC", linewidth=1.0, linestyle="-", zorder=1, label="y = x")

    # Puntos por nivel (de menor a mayor para que idéntico quede encima)
    for nv in nivel_order:
        mask = np.array([nl == nv for nl in niveles_list])
        if mask.sum() == 0:
            continue
        ax.scatter(
            tfidf_vals[mask], llm_vals[mask],
            c=nivel_colores[nv], s=18 if nv != "idéntico" else 40,
            alpha=0.7 if nv != "idéntico" else 0.95,
            edgecolors="none", label=nv, zorder=3,
        )

    # Texto de la región semántica
    n_semanticos = int(((tfidf_vals < 0.40) & (llm_vals >= 0.40)).sum())
    ax.text(
        0.20, 0.72,
        f"{n_semanticos} pares\ndetectados\npor semántica",
        transform=ax.transAxes, fontsize=8, color="#1565C0", alpha=0.85,
        va="top", ha="center",
    )

    ax.set_xlabel("Score TF-IDF coseno", fontsize=10)
    ax.set_ylabel("Score LLM", fontsize=10)
    ax.set_title(
        "Score híbrido TF-IDF vs. evaluación LLM\n1 505 pares candidatos — DCBI 2026",
        fontsize=11, pad=12,
    )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, markerscale=1.2, framealpha=0.0, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ruta_png = _guardar_fig(fig, SALIDA / "scatter_tfidf_llm.png")
    plt.close(fig)
    log.info("  PNG: %s", ruta_png)

    # ── HTML (Plotly) ─────────────────────────────────────────────────────────
    traces = []
    for nv in nivel_order:
        mask = [nl == nv for nl in niveles_list]
        if not any(mask):
            continue
        x_nv = [tfidf_vals[k] for k, m in enumerate(mask) if m]
        y_nv = [llm_vals[k]  for k, m in enumerate(mask) if m]
        hover = [
            f"<b>{clave_a_list[k]} × {clave_b_list[k]}</b><br>TF-IDF: {tfidf_vals[k]:.2f}<br>LLM: {llm_vals[k]:.2f}<br>Nivel: {niveles_list[k]}"
            for k, m in enumerate(mask) if m
        ]
        traces.append(go.Scatter(
            x=x_nv, y=y_nv, mode="markers",
            marker=dict(color=nivel_colores[nv], size=6 if nv != "idéntico" else 10, opacity=0.75),
            name=nv,
            text=hover, hoverinfo="text",
        ))
    # Línea y=x
    traces.append(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#CCCCCC", dash="dot", width=1),
        name="y = x", hoverinfo="skip",
    ))

    fig_pl = go.Figure(traces)
    fig_pl.add_vline(x=0.40, line=dict(color="#90CAF9", dash="dash", width=1))
    fig_pl.add_hline(y=0.40, line=dict(color="#90CAF9", dash="dash", width=1))
    fig_pl.update_layout(
        title="Score TF-IDF vs. evaluación LLM — 1 505 pares candidatos",
        xaxis_title="Score TF-IDF coseno",
        yaxis_title="Score LLM",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0D0D0D", size=11),
        legend=dict(title="Nivel", bordercolor="#EEEEEE", borderwidth=1),
        width=720, height=640,
    )
    fig_pl.update_xaxes(showgrid=True, gridcolor="#EEEEEE", range=[-0.02, 1.05])
    fig_pl.update_yaxes(showgrid=True, gridcolor="#EEEEEE", range=[-0.02, 1.05])

    ruta_html = _guardar_html(fig_pl.to_html(full_html=True, include_plotlyjs="cdn"), SALIDA / "scatter_tfidf_llm.html")
    log.info("  HTML: %s", ruta_html)

    return ruta_png, ruta_html


# ══════════════════════════════════════════════════════════════════════════════
# A3. Matriz transversal — UEAs presentes en 5+ licenciaturas
# ══════════════════════════════════════════════════════════════════════════════

def fig_A3_matriz_transversal(corpus, mat: np.ndarray) -> tuple[Path, Path]:
    """Heatmap: UEAs que aparecen en >= 5 licenciaturas (por similitud >= 0.70)."""
    import plotly.graph_objects as go
    from matplotlib.colors import LinearSegmentedColormap

    _setup_rcparams()
    n = len(corpus)

    # Para cada UEA i, encontrar a qué licenciaturas tiene al menos un par >= 0.70
    # (incluyendo su propia licenciatura con score 1.0 intra-plan)
    # Estrategia: para cada UEA, calcular el score máximo hacia cada licenciatura

    lics = [l for l in ORDEN if any(cu.licenciatura == l for cu in corpus)]
    idx_por_lic: dict[str, list[int]] = defaultdict(list)
    for i, cu in enumerate(corpus):
        idx_por_lic[cu.licenciatura].append(i)

    # Matriz: rows = UEAs, cols = licenciaturas, val = max sim hacia esa lic
    # (diagonal = 1.0 para propia licenciatura)
    n_lics = len(lics)
    presencia = np.zeros((n, n_lics), dtype=np.float32)
    for i, cu in enumerate(corpus):
        for jl, lic in enumerate(lics):
            if lic == cu.licenciatura:
                presencia[i, jl] = 1.0
            else:
                idxs_lic = idx_por_lic[lic]
                if idxs_lic:
                    presencia[i, jl] = float(mat[i, idxs_lic].max())

    # Contar en cuántas licenciaturas aparece cada UEA (score >= 0.70)
    n_lics_presentes = (presencia >= 0.70).sum(axis=1)
    # Filtrar UEAs en 5+ licenciaturas
    mask_5plus = n_lics_presentes >= 5
    idxs_5plus = np.where(mask_5plus)[0]

    log.info("  UEAs en 5+ licenciaturas: %d", len(idxs_5plus))

    if len(idxs_5plus) == 0:
        # Bajar umbral a 4
        mask_5plus = n_lics_presentes >= 4
        idxs_5plus = np.where(mask_5plus)[0]
        log.info("  (umbral bajado a 4) UEAs: %d", len(idxs_5plus))

    # Ordenar por número de licenciaturas (desc)
    order = sorted(idxs_5plus, key=lambda i: -n_lics_presentes[i])
    sub = presencia[order, :]
    nombres = [
        f"{corpus[i].clave} ({corpus[i].licenciatura[:3]})"
        for i in order
    ]
    n_rows = len(order)

    # ── PNG ───────────────────────────────────────────────────────────────────
    # Colormap blanco → uamred
    cmap_custom = LinearSegmentedColormap.from_list(
        "uam", ["#FFFFFF", "#FFCDD2", ACCENT], N=256
    )

    cell_h = max(0.20, min(0.40, 14.0 / n_rows))
    fig_h = max(6, n_rows * cell_h + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h), facecolor="white")

    im = ax.imshow(sub, cmap=cmap_custom, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Score máximo de similitud")

    ax.set_xticks(range(n_lics))
    ax.set_xticklabels(
        [l[:6] for l in lics], rotation=45, ha="right", fontsize=8
    )
    if n_rows <= 60:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(nombres, fontsize=6, fontfamily="monospace")
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"UEAs con presencia en ≥ 5 licenciaturas ({n_rows} UEAs)\n"
        "Valor = score máximo de similitud hacia cada licenciatura",
        fontsize=10, pad=12,
    )

    ruta_png = _guardar_fig(fig, SALIDA / "matriz_transversal.png")
    plt.close(fig)
    log.info("  PNG: %s", ruta_png)

    # ── HTML (Plotly) ─────────────────────────────────────────────────────────
    # Etiquetas cortas para hover
    hover_text = [
        [f"<b>{nombres[r]}</b><br>{lics[c]}<br>Score: {sub[r, c]:.2f}"
         for c in range(n_lics)]
        for r in range(n_rows)
    ]

    fig_pl = go.Figure(go.Heatmap(
        z=sub.tolist(),
        x=lics,
        y=nombres,
        colorscale=[[0, "#FFFFFF"], [0.5, "#FFCDD2"], [1, ACCENT]],
        zmin=0, zmax=1,
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(title="Score"),
    ))
    fig_pl.update_layout(
        title=f"UEAs con presencia en ≥ 5 licenciaturas ({n_rows} UEAs)",
        xaxis_title="Licenciatura",
        yaxis_title="UEA",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0D0D0D", size=10),
        height=max(500, n_rows * 14 + 150),
        width=700,
        yaxis=dict(autorange="reversed"),
    )

    ruta_html = _guardar_html(fig_pl.to_html(full_html=True, include_plotlyjs="cdn"), SALIDA / "matriz_transversal.html")
    log.info("  HTML: %s", ruta_html)

    return ruta_png, ruta_html


# ══════════════════════════════════════════════════════════════════════════════
# A4. Histograma / KDE intra-plan por licenciatura
# ══════════════════════════════════════════════════════════════════════════════

def fig_A4_histograma_intraplan(corpus, mat: np.ndarray) -> tuple[Path, Path]:
    """KDE superpuesto y facet 2×5 para pares intra-plan con score >= 0.30."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde

    _setup_rcparams()
    n = len(corpus)

    idx_por_lic: dict[str, list[int]] = defaultdict(list)
    for i, cu in enumerate(corpus):
        idx_por_lic[cu.licenciatura].append(i)

    lics_validas = [l for l in ORDEN if len(idx_por_lic[l]) >= 3]

    # Extraer scores intra-plan >= 0.30 (fuera de diagonal)
    scores_por_lic: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}
    for lic in lics_validas:
        idxs = idx_por_lic[lic]
        sub = mat[np.ix_(idxs, idxs)].copy()
        np.fill_diagonal(sub, 0.0)
        scores = sub[sub >= 0.30].flatten()
        scores_por_lic[lic] = scores
        medians[lic] = float(np.median(scores)) if len(scores) > 0 else 0.0

    # ── PNG (facet 2×5) ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False, facecolor="white")
    axes_flat = axes.flatten()
    x_grid = np.linspace(0.30, 1.0, 200)

    for k, lic in enumerate(lics_validas[:10]):
        ax = axes_flat[k]
        scores = scores_por_lic[lic]
        color = _c(lic)

        if len(scores) >= 5:
            try:
                kde = gaussian_kde(scores, bw_method=0.2)
                y_kde = kde(x_grid)
                ax.fill_between(x_grid, y_kde, alpha=0.25, color=color)
                ax.plot(x_grid, y_kde, color=color, linewidth=1.5)
            except Exception:
                ax.hist(scores, bins=15, density=True, color=color, alpha=0.5)
        else:
            ax.hist(scores, bins=8, density=True, color=color, alpha=0.5)

        med = medians[lic]
        ax.axvline(med, color=color, linewidth=1.2, linestyle="--", alpha=0.9)
        ax.text(
            med + 0.01, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
            f"med={med:.2f}", fontsize=7, color=color, va="top",
        )
        ax.set_title(lic, fontsize=9, color=color, pad=4)
        ax.set_xlim(0.28, 1.02)
        ax.tick_params(labelsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Score", fontsize=7)

    # Ocultar subplots vacíos si < 10 licenciaturas
    for k in range(len(lics_validas), 10):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        "Distribución de similitud intra-plan por licenciatura (s ≥ 0.30)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout(pad=1.0)

    ruta_png = _guardar_fig(fig, SALIDA / "histograma_intraplan.png")
    plt.close(fig)
    log.info("  PNG: %s", ruta_png)

    # ── HTML (Plotly — subplots 2×5) ──────────────────────────────────────────
    rows, cols = 2, 5
    fig_pl = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=lics_validas[:10],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )

    for k, lic in enumerate(lics_validas[:10]):
        row = k // cols + 1
        col = k % cols + 1
        scores = scores_por_lic[lic]
        color = _c(lic)
        med = medians[lic]

        if len(scores) >= 5:
            try:
                kde = gaussian_kde(scores, bw_method=0.2)
                x_pl = np.linspace(0.28, 1.02, 150)
                y_pl = kde(x_pl)
                fig_pl.add_trace(
                    go.Scatter(
                        x=x_pl.tolist(), y=y_pl.tolist(),
                        fill="tozeroy", mode="lines",
                        line=dict(color=color, width=2),
                        fillcolor=color.replace("#", "rgba(") + ",0.2)" if len(color) == 7
                            else color,
                        name=lic, showlegend=False,
                        hovertemplate=f"<b>{lic}</b><br>Score: %{{x:.2f}}<extra></extra>",
                    ),
                    row=row, col=col,
                )
            except Exception:
                pass
        # Línea mediana
        fig_pl.add_vline(
            x=med, line=dict(color=color, dash="dash", width=1.5),
            row=row, col=col,
            annotation_text=f"med={med:.2f}",
            annotation_font_size=9,
            annotation_position="top right",
        )

    fig_pl.update_layout(
        title="Distribución de similitud intra-plan por licenciatura (s ≥ 0.30)",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0D0D0D", size=10),
        height=520, width=1100,
        showlegend=False,
    )
    for axis in fig_pl.layout:
        if axis.startswith("xaxis"):
            fig_pl.layout[axis].update(range=[0.28, 1.02], showgrid=True, gridcolor="#EEEEEE")
        if axis.startswith("yaxis"):
            fig_pl.layout[axis].update(showgrid=False)

    ruta_html = _guardar_html(fig_pl.to_html(full_html=True, include_plotlyjs="cdn"), SALIDA / "histograma_intraplan.html")
    log.info("  HTML: %s", ruta_html)

    return ruta_png, ruta_html


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("  Generación de figuras nuevas — DCBI 2026")
    log.info("=" * 60)

    corpus, mat_hibrida, mat_tfidf, pares_llm = cargar_datos()

    resultados: list[tuple[str, bool, str]] = []

    # ── A1. Barras interplan ──────────────────────────────────────────────────
    log.info("[A1] barras_interplan_top.png + .html ...")
    try:
        png, html = fig_A1_barras_interplan(corpus, mat_hibrida)
        log.info("  OK: %s, %s", png.name, html.name)
        resultados.append(("A1", True, ""))
    except Exception as e:
        log.exception("ERROR en A1")
        resultados.append(("A1", False, str(e)))

    # ── A2. Scatter TF-IDF vs LLM ─────────────────────────────────────────────
    log.info("[A2] scatter_tfidf_llm.png + .html ...")
    try:
        png, html = fig_A2_scatter_tfidf_llm(corpus, mat_hibrida, mat_tfidf, pares_llm)
        log.info("  OK: %s, %s", png.name, html.name)
        resultados.append(("A2", True, ""))
    except Exception as e:
        log.exception("ERROR en A2")
        resultados.append(("A2", False, str(e)))

    # ── A3. Matriz transversal ────────────────────────────────────────────────
    log.info("[A3] matriz_transversal.png + .html ...")
    try:
        png, html = fig_A3_matriz_transversal(corpus, mat_hibrida)
        log.info("  OK: %s, %s", png.name, html.name)
        resultados.append(("A3", True, ""))
    except Exception as e:
        log.exception("ERROR en A3")
        resultados.append(("A3", False, str(e)))

    # ── A4. Histograma intraplan ──────────────────────────────────────────────
    log.info("[A4] histograma_intraplan.png + .html ...")
    try:
        png, html = fig_A4_histograma_intraplan(corpus, mat_hibrida)
        log.info("  OK: %s, %s", png.name, html.name)
        resultados.append(("A4", True, ""))
    except Exception as e:
        log.exception("ERROR en A4")
        resultados.append(("A4", False, str(e)))

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    ok   = [r for r in resultados if r[1]]
    fail = [r for r in resultados if not r[1]]
    log.info("Exitosas: %d/4", len(ok))
    if fail:
        log.warning("Fallidas: %d/4", len(fail))
        for nombre, _, err in fail:
            log.warning("  %s: %s", nombre, err)
    log.info("Figuras en: %s", SALIDA)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
