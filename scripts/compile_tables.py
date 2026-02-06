import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABEL = {
    "random": "Amostragem Aleatoria",
    "us_single": "Incerteza (passagem unica)",
    "proposed_mc": "Proposta (MC Dropout)",
    "Amostragem Aleatoria": "Amostragem Aleatoria",
    "Incerteza (passagem unica)": "Incerteza (passagem unica)",
    "Proposta (MC Dropout)": "Proposta (MC Dropout)",
}


def fmt_map(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}".replace(".", ",")


def fmt_auc(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.2f}".replace(".", ",")


def best_bold(col_vals, out_strs):
    best = None
    for v in col_vals:
        if pd.isna(v):
            continue
        v = float(v)
        best = v if best is None else max(best, v)
    if best is None:
        return out_strs
    out = []
    for v, s in zip(col_vals, out_strs):
        if pd.isna(v):
            out.append(s)
        else:
            out.append(f"\\textbf{{{s}}}" if np.isclose(float(v), best) else s)
    return out


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise ValueError("CSV de entrada precisa ter a coluna method")
    df["method"] = df["method"].map(lambda x: METHOD_LABEL.get(str(x), str(x)))
    return df


def group_keys(df: pd.DataFrame, keys: list[str]) -> list[str]:
    return [k for k in keys if k in df.columns]


def make_auc_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    need = ["method", "auc_map50", "auc_map50_norm", "auc_map50_95", "auc_map50_95_norm"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Faltando coluna {c} no CSV")
    gkeys = group_keys(df, keys)

    if gkeys:
        agg = df.groupby(gkeys + ["method"], as_index=False).agg(
            AUC_mAP50=("auc_map50", "mean"),
            mAP50_medio=("auc_map50_norm", "mean"),
            AUC_mAP50_95=("auc_map50_95", "mean"),
            mAP50_95_medio=("auc_map50_95_norm", "mean"),
            n=("method", "count"),
        )
    else:
        agg = df.groupby(["method"], as_index=False).agg(
            AUC_mAP50=("auc_map50", "mean"),
            mAP50_medio=("auc_map50_norm", "mean"),
            AUC_mAP50_95=("auc_map50_95", "mean"),
            mAP50_95_medio=("auc_map50_95_norm", "mean"),
            n=("method", "count"),
        )

    order = ["Amostragem Aleatoria", "Incerteza (passagem unica)", "Proposta (MC Dropout)"]
    agg["method"] = pd.Categorical(agg["method"], categories=order, ordered=True)
    agg = agg.sort_values(gkeys + ["method"]).reset_index(drop=True)

    return agg


def auc_table_to_latex(agg: pd.DataFrame, keys: list[str], caption: str, label: str) -> str:
    gkeys = group_keys(agg, keys)
    cols = gkeys + ["method", "AUC_mAP50", "mAP50_medio", "AUC_mAP50_95", "mAP50_95_medio"]
    t = agg[cols].copy()

    map50_s = [fmt_map(v) for v in t["mAP50_medio"].to_numpy()]
    map95_s = [fmt_map(v) for v in t["mAP50_95_medio"].to_numpy()]
    auc50_s = [fmt_auc(v) for v in t["AUC_mAP50"].to_numpy()]
    auc95_s = [fmt_auc(v) for v in t["AUC_mAP50_95"].to_numpy()]

    auc50_s = best_bold(t["AUC_mAP50"].to_numpy(), auc50_s)
    map50_s = best_bold(t["mAP50_medio"].to_numpy(), map50_s)
    auc95_s = best_bold(t["AUC_mAP50_95"].to_numpy(), auc95_s)
    map95_s = best_bold(t["mAP50_95_medio"].to_numpy(), map95_s)

    lines = []
    lines.append("\\begin{table}[!htb]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    spec_cols = "l" * (len(gkeys) + 1) + "cccc"
    lines.append("\\resizebox{\\linewidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{spec_cols}}}")
    lines.append("\\toprule")

    header_parts = []
    for k in gkeys:
        header_parts.append(f"\\textbf{{{k}}}")
    header_parts += [
        "\\textbf{Metodo}",
        "\\textbf{AUC mAP@50}",
        "\\textbf{$\\overline{\\text{mAP@50}}$}",
        "\\textbf{AUC mAP@50--95}",
        "\\textbf{$\\overline{\\text{mAP@50--95}}$}",
    ]
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for i in range(len(t)):
        row = []
        for k in gkeys:
            row.append(str(t.iloc[i][k]))
        row += [
            str(t.iloc[i]["method"]),
            auc50_s[i],
            map50_s[i],
            auc95_s[i],
            map95_s[i],
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_final_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    need = ["method", "final_map50", "final_map50_95", "final_precision", "final_recall", "final_infer_ms"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Faltando coluna {c} no CSV")

    gkeys = group_keys(df, keys)
    if gkeys:
        agg = df.groupby(gkeys + ["method"], as_index=False).agg(
            mAP50=("final_map50", "mean"),
            mAP50_95=("final_map50_95", "mean"),
            Precision=("final_precision", "mean"),
            Recall=("final_recall", "mean"),
            Infer_ms=("final_infer_ms", "mean"),
            n=("method", "count"),
        )
    else:
        agg = df.groupby(["method"], as_index=False).agg(
            mAP50=("final_map50", "mean"),
            mAP50_95=("final_map50_95", "mean"),
            Precision=("final_precision", "mean"),
            Recall=("final_recall", "mean"),
            Infer_ms=("final_infer_ms", "mean"),
            n=("method", "count"),
        )

    order = ["Amostragem Aleatoria", "Incerteza (passagem unica)", "Proposta (MC Dropout)"]
    agg["method"] = pd.Categorical(agg["method"], categories=order, ordered=True)
    return agg.sort_values(gkeys + ["method"]).reset_index(drop=True)


def final_table_to_latex(agg: pd.DataFrame, keys: list[str], caption: str, label: str) -> str:
    gkeys = group_keys(agg, keys)
    cols = gkeys + ["method", "mAP50", "mAP50_95", "Precision", "Recall", "Infer_ms"]
    t = agg[cols].copy()

    m50_s = [fmt_map(v) for v in t["mAP50"].to_numpy()]
    m95_s = [fmt_map(v) for v in t["mAP50_95"].to_numpy()]
    p_s = [fmt_map(v) for v in t["Precision"].to_numpy()]
    r_s = [fmt_map(v) for v in t["Recall"].to_numpy()]
    inf_s = [("-" if pd.isna(v) else f"{float(v):.1f}".replace(".", ",")) for v in t["Infer_ms"].to_numpy()]

    m50_s = best_bold(t["mAP50"].to_numpy(), m50_s)
    m95_s = best_bold(t["mAP50_95"].to_numpy(), m95_s)
    p_s = best_bold(t["Precision"].to_numpy(), p_s)
    r_s = best_bold(t["Recall"].to_numpy(), r_s)

    lines = []
    lines.append("\\begin{table}[!htb]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    spec_cols = "l" * (len(gkeys) + 1) + "cccc" + "c"
    lines.append("\\resizebox{\\linewidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{spec_cols}}}")
    lines.append("\\toprule")

    header_parts = []
    for k in gkeys:
        header_parts.append(f"\\textbf{{{k}}}")
    header_parts += [
        "\\textbf{Metodo}",
        "\\textbf{mAP@50}",
        "\\textbf{mAP@50--95}",
        "\\textbf{Precisao}",
        "\\textbf{Recall}",
        "\\textbf{Infer (ms)}",
    ]
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for i in range(len(t)):
        row = []
        for k in gkeys:
            row.append(str(t.iloc[i][k]))
        row += [
            str(t.iloc[i]["method"]),
            m50_s[i],
            m95_s[i],
            p_s[i],
            r_s[i],
            inf_s[i],
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="runs/all_results_compiled.csv")
    ap.add_argument("--out_dir", type=str, default="tables_out")
    ap.add_argument("--group_by", type=str, default="l0,budget,cycles")
    return ap.parse_args()


def main():
    a = parse_args()
    inp = Path(a.input)
    if not inp.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {inp}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [k.strip() for k in a.group_by.split(",") if k.strip()]
    df = load_input(inp)

    auc_tbl = make_auc_table(df, keys)
    auc_tbl.to_csv(out_dir / "auc_table.csv", index=False)

    latex_auc = auc_table_to_latex(
        auc_tbl,
        keys,
        caption="Analise de custo-beneficio via AUC de desempenho versus amostras rotuladas (0 a 400). A AUC normalizada corresponde ao desempenho medio ao longo do orcamento de anotacao.",
        label="tab:auc_cost_benefit",
    )
    (out_dir / "auc_table.tex").write_text(latex_auc, encoding="utf-8")

    final_tbl = make_final_table(df, keys)
    final_tbl.to_csv(out_dir / "final_metrics_table.csv", index=False)

    latex_final = final_table_to_latex(
        final_tbl,
        keys,
        caption="Desempenho final ao termino dos ciclos (media por configuracao).",
        label="tab:final_metrics",
    )
    (out_dir / "final_metrics_table.tex").write_text(latex_final, encoding="utf-8")

    print(f"Gerado: {out_dir / 'auc_table.csv'}")
    print(f"Gerado: {out_dir / 'auc_table.tex'}")
    print(f"Gerado: {out_dir / 'final_metrics_table.csv'}")
    print(f"Gerado: {out_dir / 'final_metrics_table.tex'}")


if __name__ == "__main__":
    main()
