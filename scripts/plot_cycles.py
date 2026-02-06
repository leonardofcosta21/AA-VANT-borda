import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROTULO_METODO = {
    "random": "Amostragem Aleatória",
    "us_single": "Incerteza (passagem única)",
    "proposed_mc": "Proposta (MC Dropout)",
}

ROTULO_METRICA = {
    "mAP50": "mAP@50",
    "mAP50_95": "mAP@50–95",
}

MARCADOR = {
    "proposed_mc": "o",
    "us_single": "^",
    "random": "s",
}

OFFSET_TEXTO = {
    "proposed_mc": +0.010,
    "us_single":   0.000,
    "random":     -0.010,
}

ORDEM = ["proposed_mc", "us_single", "random"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="plots_final")
    ap.add_argument("--metrica", type=str, default="mAP50", choices=["mAP50", "mAP50_95"])
    ap.add_argument("--x", type=str, default="cycle", choices=["cycle", "labeled_count"])
    ap.add_argument("--titulo", type=str, default="")
    ap.add_argument("--dpi", type=int, default=250)
    return ap.parse_args()


def ler_results(exp_dir: Path) -> pd.DataFrame:
    arquivos = sorted(exp_dir.glob("*_seed*/results.csv"))
    partes = []
    for p in arquivos:
        metodo = p.parent.name.split("_seed")[0]
        df = pd.read_csv(p)
        df["method"] = metodo
        partes.append(df)
    if not partes:
        raise FileNotFoundError(f"Nenhum results.csv encontrado em: {exp_dir}")
    return pd.concat(partes, ignore_index=True)


def fmt_virgula(x: float, casas: int = 4) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def plot(df: pd.DataFrame, metrica: str, x: str, titulo: str, saida: Path, dpi: int = 250):
    fig, ax = plt.subplots()

    medias = {}
    cores = {}

    for metodo in ORDEM:
        dd = df[df["method"] == metodo].copy()
        if dd.empty or metrica not in dd.columns or x not in dd.columns:
            continue

        dd = dd.sort_values(x)
        y = dd[metrica].to_numpy(dtype=float)
        xv = dd[x].to_numpy(dtype=float)
        if len(xv) == 0:
            continue

        (linha,) = ax.plot(
            xv,
            y,
            marker=MARCADOR.get(metodo, "o"),
            markersize=5,
            linewidth=1.8,
            label=ROTULO_METODO.get(metodo, metodo),
        )

        cor = linha.get_color()
        cores[metodo] = cor

        m = float(np.nanmean(y))
        medias[metodo] = m

        ax.hlines(
            m,
            xmin=float(np.nanmin(xv)),
            xmax=float(np.nanmax(xv)),
            linestyles="dotted",
            linewidth=1.6,
            color=cor,
            alpha=0.9,
        )

    if not medias:
        raise ValueError("Não foi possível plotar: faltam dados para os métodos.")

    ys = np.array(list(medias.values()), dtype=float)
    y_range = float(np.nanmax(ys) - np.nanmin(ys))
    y_step = max(0.012, 0.06 * y_range) if y_range > 0 else 0.012

    itens = [(m, v) for m, v in medias.items() if np.isfinite(v)]
    itens.sort(key=lambda t: t[1])

    y_pos = {}
    ultimo = None
    for metodo, yv in itens:
        y_adj = yv if ultimo is None else max(yv, ultimo + y_step)
        y_pos[metodo] = y_adj
        ultimo = y_adj

    for metodo in ORDEM:
        if metodo not in medias:
            continue
        y_texto = y_pos[metodo] + OFFSET_TEXTO.get(metodo, 0.0)

        ax.text(
            1.01,
            y_texto,
            f"média={fmt_virgula(medias[metodo], 4)}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10,
            color=cores.get(metodo, "black"),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
            clip_on=False,
        )


    ax.set_xlabel("Ciclo" if x == "cycle" else "Amostras rotuladas")
    ax.set_ylabel(ROTULO_METRICA.get(metrica, metrica))
    if titulo:
        ax.set_title(titulo)

    ax.legend(loc="lower right")
    fig.tight_layout()
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    exp_dir = Path(a.exp_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ler_results(exp_dir)

    base = exp_dir.name
    base = base.replace("seed42__", "").replace("__seed42", "")
    base = base.replace("seed", "")
    nome = f"{base}__{a.metrica}_por_{a.x}_com_medias.png"

    saida = out_dir / nome

    titulo = a.titulo.strip()
    if not titulo:
        titulo = f"{ROTULO_METRICA.get(a.metrica, a.metrica)} ao longo de {('ciclos' if a.x=='cycle' else 'rótulos')}"

    plot(df, a.metrica, a.x, titulo, saida, a.dpi)
    print(f"Figura salva em: {saida.resolve()}")


if __name__ == "__main__":
    main()
