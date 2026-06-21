"""
generar_grafo_mejorado.py
Genera dos variantes de visualización de traslape inter-licenciaturas DCBI 2026.
Variante A: Chord diagram (matplotlib puro, curvas Bézier)
Variante B: Network graph filtrado (networkx, Kamada-Kawai)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as patches
import networkx as nx
import os

# ── Directorios ──────────────────────────────────────────────────────────────
BASE = "/Users/nowhere/Claude_Projects/Admin_duties/Modificaciones2026/Revisiones"
FIGDIR = os.path.join(BASE, "Figuras")

# ── Datos ────────────────────────────────────────────────────────────────────
TICS = ["Ambiental", "Civil", "Computación", "Eléctrica",
        "Electrónica", "Física", "Industrial", "Mecánica",
        "Metalúrgica", "Química"]

N_UEAS = {
    "Ambiental": 40, "Civil": 88, "Computación": 60, "Eléctrica": 39,
    "Electrónica": 56, "Física": 60, "Industrial": 78, "Mecánica": 108,
    "Metalúrgica": 42, "Química": 55,
}

# Matriz simétrica de conteo (i,j) = UEAs equivalentes con s≥0.70
RAW = [
    [ 0,  4,  1,  1,  1,  5,  6,  6,  2, 14],
    [ 4,  0,  6,  4,  7,  9,  9, 16,  3,  8],
    [ 1,  6,  0,  7, 12,  8,  4,  3,  0,  1],
    [ 1,  4,  7,  0,  9,  7,  5, 13,  4,  4],
    [ 1,  7, 12,  9,  0, 10, 11, 14,  2,  2],
    [ 5,  9,  8,  7, 10,  0, 14, 15,  4,  9],
    [ 6,  9,  4,  5, 11, 14,  0, 38,  6, 13],
    [ 6, 16,  3, 13, 14, 15, 38,  0, 10, 14],
    [ 2,  3,  0,  4,  2,  4,  6, 10,  0,  5],
    [14,  8,  1,  4,  2,  9, 13, 14,  5,  0],
]
M = np.array(RAW, dtype=float)
n = len(TICS)

COLORES = {
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
COLS = [COLORES[t] for t in TICS]

# Fuente sans-serif preferida
plt.rcParams["font.family"] = "DejaVu Sans"

# ════════════════════════════════════════════════════════════════════════════
# VARIANTE A: CHORD DIAGRAM
# ════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def blend_colors(c1, c2, alpha=0.5):
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return ((r1+r2)/2, (g1+g2)/2, (b1+b2)/2, alpha)

def cubic_bezier(p0, p1, p2, p3, t):
    """Curva Bézier cúbica vectorizada."""
    t = np.atleast_1d(t)[:, None]
    return ((1-t)**3 * p0 + 3*(1-t)**2*t * p1 +
            3*(1-t)*t**2 * p2 + t**3 * p3)

def chord_arc_path(theta1, theta2, r_inner=0.75, n_pts=60):
    """
    Genera los vértices de una 'cuerda' entre dos arcos angulares
    usando curvas Bézier cúbicas que pasan por el centro.
    """
    # Puntos de inicio/fin en el borde interior del anillo
    p0 = np.array([r_inner * np.cos(theta1), r_inner * np.sin(theta1)])
    p3 = np.array([r_inner * np.cos(theta2), r_inner * np.sin(theta2)])
    # Puntos de control: hacia el centro
    ctrl_scale = 0.3
    p1 = p0 * ctrl_scale
    p2 = p3 * ctrl_scale
    t_vals = np.linspace(0, 1, n_pts)
    pts = cubic_bezier(p0, p1, p2, p3, t_vals)
    return pts

def arc_points(theta_start, theta_end, r=1.0, n=60):
    thetas = np.linspace(theta_start, theta_end, n)
    return np.stack([r*np.cos(thetas), r*np.sin(thetas)], axis=1)

def make_chord_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor="white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")
    ax.set_facecolor("white")

    # Suma total por fila (flujo saliente de cada licenciatura)
    row_sums = M.sum(axis=1)
    total = row_sums.sum()

    # Gap angular entre segmentos (en radianes)
    GAP = 0.03   # radianes entre segmentos
    total_gap = GAP * n
    total_data_angle = 2*np.pi - total_gap

    # Ángulo asignado a cada licenciatura ∝ suma de su fila
    seg_angles = (row_sums / total) * total_data_angle

    # Calcular inicio y fin de cada segmento (empezando en -π/2, sentido antihorario)
    seg_start = np.zeros(n)
    seg_end   = np.zeros(n)
    cursor = -np.pi/2
    for i in range(n):
        seg_start[i] = cursor
        seg_end[i]   = cursor + seg_angles[i]
        cursor = seg_end[i] + GAP

    # ── Radio del anillo externo
    R_OUT  = 1.05
    R_IN   = 0.90
    R_TICK = 1.12   # para etiquetas

    # ── Dibujar cuerdas (de menor a mayor para que las grandes queden arriba)
    # Recopilar todos los pares con su peso
    chord_list = []
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] > 0:
                chord_list.append((M[i, j], i, j))
    chord_list.sort()  # orden ascendente → las grandes se dibujan al final

    # Acumuladores de ángulo dentro de cada segmento para las cuerdas
    seg_used_start = seg_start.copy()
    seg_used = np.zeros(n)   # cuánto se ha consumido del ángulo de cada seg

    # Pre-calcular sub-ángulos para cada par (i,j)
    # La cuerda (i,j) ocupa una fracción del segmento i proporcional a M[i,j]/row_sums[i]
    # y análogamente en j
    chord_angles = {}
    seg_cursor = seg_start.copy()
    for i in range(n):
        row_i = [(j, M[i,j]) for j in range(n) if j != i and M[i,j] > 0]
        # ordenar por j para reproducibilidad
        row_i.sort(key=lambda x: x[0])
        for j, val in row_i:
            frac = val / row_sums[i] * seg_angles[i]
            chord_angles[(i, j)] = (seg_cursor[i], seg_cursor[i] + frac)
            seg_cursor[i] += frac

    MAX_W = 38.0
    MIN_W = 0.5

    for (val, i, j) in chord_list:
        # Ángulos medios de la porción de la cuerda en cada segmento
        ai_s, ai_e = chord_angles[(i, j)]
        aj_s, aj_e = chord_angles[(j, i)]
        ai_mid = (ai_s + ai_e) / 2
        aj_mid = (aj_s + aj_e) / 2

        # Ancho de la cuerda ∝ val, mapeado a grosor visual
        lw = MIN_W + (val / MAX_W) * 9.5

        # Color blended con alfa ∝ peso
        alpha = 0.25 + 0.60 * (val / MAX_W)
        color = blend_colors(COLS[i], COLS[j], alpha)

        # Puntos de la curva Bézier
        pts = chord_arc_path(ai_mid, aj_mid, r_inner=0.88, n_pts=80)
        ax.plot(pts[:, 0], pts[:, 1], lw=lw, color=color,
                solid_capstyle="round", zorder=2)

    # ── Arcos externos (uno por licenciatura)
    for i in range(n):
        theta_s = seg_start[i]
        theta_e = seg_end[i]
        thetas = np.linspace(theta_s, theta_e, 100)

        # Anillo exterior (relleno)
        inner = arc_points(theta_s, theta_e, R_IN)
        outer = arc_points(theta_e, theta_s, R_OUT)  # invertido para cerrar
        verts = np.vstack([inner, outer])
        codes = ([Path.MOVETO] + [Path.LINETO]*(len(inner)-1) +
                 [Path.LINETO] + [Path.LINETO]*(len(outer)-1) +
                 [Path.CLOSEPOLY])
        # Simplificar: usar fill_between polar via plot con ancho de línea grueso
        ax.plot(R_OUT * np.cos(thetas), R_OUT * np.sin(thetas),
                color=COLS[i], lw=10, solid_capstyle="butt", zorder=5)
        ax.plot(R_IN * np.cos(thetas), R_IN * np.sin(thetas),
                color=COLS[i], lw=10, solid_capstyle="butt", zorder=5, alpha=0.4)
        # Relleno del arco (polígono)
        all_thetas = np.linspace(theta_s, theta_e, 80)
        xo = R_OUT * np.cos(all_thetas)
        yo = R_OUT * np.sin(all_thetas)
        xi = R_IN  * np.cos(all_thetas[::-1])
        yi = R_IN  * np.sin(all_thetas[::-1])
        xpoly = np.concatenate([xo, xi])
        ypoly = np.concatenate([yo, yi])
        ax.fill(xpoly, ypoly, color=COLS[i], alpha=0.9, zorder=4)

        # Etiqueta
        mid_theta = (theta_s + theta_e) / 2
        lx = R_TICK * np.cos(mid_theta)
        ly = R_TICK * np.sin(mid_theta)
        angle_deg = np.degrees(mid_theta)
        # Ajustar alineación según posición
        ha = "left" if np.cos(mid_theta) > 0 else "right"
        if abs(np.cos(mid_theta)) < 0.3:
            ha = "center"
        va = "bottom" if np.sin(mid_theta) > 0.3 else ("top" if np.sin(mid_theta) < -0.3 else "center")

        # Rotación radial
        rot = angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180
        label = TICS[i]
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=8.5, fontweight="bold",
                color=COLS[i], rotation=rot, rotation_mode="anchor", zorder=10)

    # ── Anotar la cuerda dominante Industrial-Mecánica
    i_ind = TICS.index("Industrial")
    i_mec = TICS.index("Mecánica")
    ai_mid = sum(chord_angles[(i_ind, i_mec)]) / 2
    aj_mid = sum(chord_angles[(i_mec, i_ind)]) / 2
    mid_pt = chord_arc_path(ai_mid, aj_mid, r_inner=0.88, n_pts=80)[40]
    ax.annotate("38 UEAs\n(Industrial–Mecánica)",
                xy=mid_pt, xytext=(mid_pt[0]+0.28, mid_pt[1]+0.10),
                fontsize=7.5, color="#37474F", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#37474F", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#37474F", lw=0.6),
                zorder=20)

    # ── Título
    ax.set_title("Traslape de UEAs equivalentes entre licenciaturas DCBI\n"
                 "Chord diagram — ancho de cuerda ∝ conteo (s ≥ 0.70)",
                 fontsize=10, fontweight="bold", color="#1a1a1a", pad=12)

    # ── Leyenda (parches de color)
    legend_patches = [mpatches.Patch(color=COLORES[t], label=t) for t in TICS]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              columnspacing=0.8, handlelength=1.0, handletextpad=0.4)

    plt.tight_layout(pad=0.5)
    out = os.path.join(FIGDIR, "grafo_licenciaturas_variante_A.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Variante A guardada: {out}")


# ════════════════════════════════════════════════════════════════════════════
# VARIANTE B: NETWORK GRAPH FILTRADO
# ════════════════════════════════════════════════════════════════════════════

def make_network_graph():
    UMBRAL = 5   # solo aristas con count ≥ UMBRAL

    # Construir grafo
    G = nx.Graph()
    for t in TICS:
        G.add_node(t)
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] >= UMBRAL:
                G.add_edge(TICS[i], TICS[j], weight=float(M[i, j]))

    # Layout Kamada-Kawai con pesos (distancia ∝ 1/peso)
    # Usar 1/(w+1) como distancia para que pares similares queden juntos
    dist_dict = {}
    for u in G.nodes():
        dist_dict[u] = {}
        for v in G.nodes():
            if u == v:
                dist_dict[u][v] = 0
            elif G.has_edge(u, v):
                w = G[u][v]["weight"]
                dist_dict[u][v] = 1.0 / (w + 1)
            else:
                dist_dict[u][v] = 2.0   # nodos no conectados: lejos

    try:
        pos = nx.kamada_kawai_layout(G, dist=dist_dict, scale=2.5)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=1.8, iterations=200, weight="weight")

    # ── Escalar nodos ∝ sqrt(n_ueas)
    max_ueas = max(N_UEAS.values())
    node_sizes = {t: (N_UEAS[t] / max_ueas)**0.5 * 1800 for t in TICS}

    # ── Escalar aristas: 1px @ count=5, 12px @ count=38
    w_min, w_max = UMBRAL, 38.0
    lw_min, lw_max = 1.0, 12.0
    edge_widths = []
    edge_colors = []
    edge_list = list(G.edges(data=True))
    for u, v, d in edge_list:
        w = d["weight"]
        lw = lw_min + (w - w_min) / (w_max - w_min) * (lw_max - lw_min)
        lw = max(lw_min, min(lw_max, lw))
        edge_widths.append(lw)
        # Color de la arista: color del nodo con mayor contribución
        # (mayor suma de fila = mayor "hub")
        rs_u = M[TICS.index(u)].sum()
        rs_v = M[TICS.index(v)].sum()
        if rs_u >= rs_v:
            edge_colors.append(COLORES[u])
        else:
            edge_colors.append(COLORES[v])

    # ── Figura
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Dibujar aristas
    for idx, (u, v, d) in enumerate(edge_list):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1],
                color=edge_colors[idx], lw=edge_widths[idx],
                alpha=0.65, solid_capstyle="round", zorder=1)

    # Dibujar nodos
    for t in TICS:
        x, y = pos[t]
        sz = node_sizes[t]
        radius_pts = (sz / np.pi)**0.5   # radio aproximado en puntos de scatter
        sc = ax.scatter(x, y, s=sz, color=COLORES[t],
                        edgecolors="white", linewidths=2.5,
                        zorder=3)

    # ── Etiquetas FUERA de los nodos
    # Desplazamiento radial desde el centroide de la nube de nodos
    centroid = np.mean(np.array(list(pos.values())), axis=0)
    label_offset = 0.45   # unidades de layout

    for t in TICS:
        x, y = pos[t]
        dx, dy = x - centroid[0], y - centroid[1]
        norm = (dx**2 + dy**2)**0.5 + 1e-9
        ux, uy = dx/norm, dy/norm

        # Escala del offset según tamaño del nodo (nodos grandes necesitan más espacio)
        sz_frac = (N_UEAS[t] / max_ueas)**0.5
        off = label_offset + sz_frac * 0.25

        lx, ly = x + ux * off, y + uy * off

        ha = "left" if ux > 0.2 else ("right" if ux < -0.2 else "center")
        va = "bottom" if uy > 0.15 else ("top" if uy < -0.15 else "center")

        # Línea de conexión nodo→etiqueta (thin)
        ax.annotate("", xy=(x, y), xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="-", color=COLORES[t],
                                   lw=0.8, alpha=0.6),
                    zorder=2)

        # Texto con sombra suave
        txt = ax.text(lx, ly, t, ha=ha, va=va, fontsize=9,
                      fontweight="bold", color=COLORES[t], zorder=5)
        txt.set_path_effects([
            pe.withStroke(linewidth=2.5, foreground="white")
        ])

    # ── Anotar el par dominante Industrial-Mecánica
    xi, yi = pos["Industrial"]
    xm, ym = pos["Mecánica"]
    xmid, ymid = (xi+xm)/2, (yi+ym)/2
    ax.text(xmid, ymid + 0.10, "38", fontsize=10, fontweight="bold",
            ha="center", va="bottom", color="#C62828", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#C62828", lw=1.2, alpha=0.92))

    # ── Leyenda de grosor de aristas (escala)
    scale_x = 0.02  # en ejes de figura (fracción)
    scale_y = 0.18
    scale_entries = [(5, lw_min), (16, lw_min + (16-w_min)/(w_max-w_min)*(lw_max-lw_min)), (38, lw_max)]
    legend_ax = fig.add_axes([0.02, 0.04, 0.18, 0.20], facecolor="white")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.text(0.5, 0.93, "Aristas\n(UEAs equiv.)", ha="center", va="top",
                   fontsize=7.5, fontweight="bold", color="#444444",
                   transform=legend_ax.transAxes)
    for k, (cnt, lw) in enumerate(scale_entries):
        yy = 0.70 - k * 0.25
        legend_ax.plot([0.05, 0.55], [yy, yy], lw=lw, color="#666666",
                       solid_capstyle="round", transform=legend_ax.transAxes)
        legend_ax.text(0.62, yy, f"{int(cnt)}", ha="left", va="center",
                       fontsize=7.5, color="#444444", transform=legend_ax.transAxes)
    legend_ax.set_facecolor("white")
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    # Borde suave
    legend_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                       ec="#cccccc", lw=0.8,
                                       transform=legend_ax.transAxes))

    # ── Leyenda de nodos (tamaño)
    size_ax = fig.add_axes([0.02, 0.27, 0.18, 0.18], facecolor="white")
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.axis("off")
    size_ax.text(0.5, 0.93, "Nodos\n(UEAs totales)", ha="center", va="top",
                 fontsize=7.5, fontweight="bold", color="#444444",
                 transform=size_ax.transAxes)
    for k, (label, ueas) in enumerate([(40, 40), (78, 78), (108, 108)]):
        sz_ex = (ueas / max_ueas)**0.5 * 1800
        yy_data = 0.65 - k * 0.28
        size_ax.scatter([0.22], [yy_data], s=sz_ex * 0.35,
                        color="#888888", edgecolors="white", linewidths=1.5,
                        transform=size_ax.transAxes, zorder=3)
        size_ax.text(0.40, yy_data, f"{ueas}", ha="left", va="center",
                     fontsize=7.5, color="#444444", transform=size_ax.transAxes)
    size_ax.set_facecolor("white")
    for spine in size_ax.spines.values():
        spine.set_visible(False)
    size_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                     ec="#cccccc", lw=0.8,
                                     transform=size_ax.transAxes))

    # ── Leyenda de colores (licenciaturas)
    legend_patches = [mpatches.Patch(color=COLORES[t], label=t) for t in TICS]
    ax.legend(handles=legend_patches, loc="upper right",
              bbox_to_anchor=(1.01, 1.01), ncol=1, fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              handlelength=1.0, handletextpad=0.5, borderpad=0.6)

    ax.set_title("Traslape de UEAs equivalentes entre licenciaturas DCBI\n"
                 "Aristas: conteo ≥ 5 · Nodo: tamaño ∝ UEAs totales · Layout Kamada-Kawai",
                 fontsize=10, fontweight="bold", color="#1a1a1a", pad=10)

    plt.tight_layout(pad=0.8)
    out = os.path.join(FIGDIR, "grafo_licenciaturas_variante_B.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Variante B guardada: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando Variante A (Chord diagram)…")
    make_chord_diagram()
    print("Generando Variante B (Network graph filtrado)…")
    make_network_graph()
    print("Listo.")
