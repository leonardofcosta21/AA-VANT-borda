import itertools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SweepCfg:
    seed: int
    l0: int
    budget: int
    cycles: int
    auc_budget: int = 400
    epochs: int = 5
    batch: int = 16
    imgsz: int = 640
    mc_passes: int = 8
    max_pool_eval: int | None = None


def run_cmd(cmd: list[str], cwd: Path):
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Falha:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )
    return p.stdout


def make_exp_id(c: SweepCfg) -> str:
    parts = [f"seed{c.seed}", f"l0_{c.l0}", f"b_{c.budget}", f"c_{c.cycles}", f"auc_{c.auc_budget}"]
    if c.max_pool_eval is not None:
        parts.append(f"poolcap_{c.max_pool_eval}")
    return "__".join(parts)


def main():
    repo = Path(__file__).resolve().parents[1]
    out_root = repo / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [42]
    l0_list = [100, 300]
    budgets = [50]
    cycles_list = [8]

    grid = [
        SweepCfg(seed=s, l0=l0, budget=b, cycles=cy)
        for s, l0, b, cy in itertools.product(seeds, l0_list, budgets, cycles_list)
    ]

    master_rows = []

    for c in grid:
        eid = make_exp_id(c)
        exp_dir = out_root / eid
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config.json").write_text(json.dumps(c.__dict__, indent=2), encoding="utf-8")

        splits_dir = repo / "splits"
        if splits_dir.exists():
            shutil.rmtree(splits_dir)

        run_cmd(
            ["python", "scripts/prepare_splits.py", "--seed", str(c.seed), "--l0", str(c.l0)],
            cwd=repo,
        )

        cmd = [
            "python",
            "scripts/run_cycles.py",
            "--seed",
            str(c.seed),
            "--cycles",
            str(c.cycles),
            "--budget",
            str(c.budget),
            "--auc_budget",
            str(c.auc_budget),
            "--epochs",
            str(c.epochs),
            "--batch",
            str(c.batch),
            "--imgsz",
            str(c.imgsz),
            "--mc_passes",
            str(c.mc_passes),
            "--out_root",
            str(exp_dir.relative_to(repo)),
        ]
        if c.max_pool_eval is not None:
            cmd += ["--max_pool_eval", str(c.max_pool_eval)]

        run_cmd(cmd, cwd=repo)

        summaries = sorted(exp_dir.glob("*_seed*/summary.csv"))
        if summaries:
            df = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
            df.insert(0, "exp_id", eid)
            df.insert(1, "l0", c.l0)
            df.insert(2, "budget", c.budget)
            df.insert(3, "cycles", c.cycles)
            if "seed" not in df.columns:
                df.insert(4, "seed", c.seed)
            df.to_csv(exp_dir / "summary_all_methods.csv", index=False)
            master_rows.append(df)

        print(f"OK: {eid}")

    if master_rows:
        pd.concat(master_rows, ignore_index=True).to_csv(out_root / "all_results_compiled.csv", index=False)
        print(f"Gerado: {out_root / 'all_results_compiled.csv'}")


if __name__ == "__main__":
    main()
