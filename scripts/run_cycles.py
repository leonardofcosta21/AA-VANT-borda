import argparse
import time
import random
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Cfg:
    seed: int = 42
    cycles: int = 8
    budget_per_cycle: int = 50
    imgsz: int = 640
    epochs_per_cycle: int = 5
    batch: int = 16
    mc_passes: int = 8
    auc_budget_labels: int = 400
    weights0: str = "yolov8n.pt"
    max_pool_eval: int | None = None
    out_root: str = "runs"
    device: str | None = None
    deterministic: bool = True


def enable_dropout_only(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout) or m.__class__.__name__.lower().startswith("dropout"):
            m.train()


def set_bn_eval(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or "batchnorm" in m.__class__.__name__.lower():
            m.eval()


def list_images(folder: Path):
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS])


def yolo_data_yaml(splits_root: Path, out_yaml: Path, sard_yaml: Path):
    base = yaml.safe_load(sard_yaml.read_text())
    data = {
        "path": str(splits_root),
        "train": "labeled/train/images",
        "val": "labeled/val/images",
        "test": "test/images",
        "nc": int(base["nc"]),
        "names": base["names"],
    }
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def reveal_label_and_move(img_path, raw_labels, labeled_train_images, labeled_train_labels):
    labeled_train_images.mkdir(parents=True, exist_ok=True)
    labeled_train_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, labeled_train_images / img_path.name)
    lbl = raw_labels / (img_path.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, labeled_train_labels / lbl.name)
    else:
        (labeled_train_labels / (img_path.stem + ".txt")).write_text("")


def uncertainty_score_mc(model, img_path, mc_passes):
    confs = []
    for _ in range(mc_passes):
        r = model.predict(source=str(img_path), imgsz=model.predictor.args.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            confs.append(0.0)
        else:
            confs.append(float(r.boxes.conf.max().item()))
    confs = np.array(confs, dtype=np.float32)
    return (1.0 - float(confs.mean())) + float(confs.std()), float(confs.mean()), float(confs.std())


def uncertainty_score_single(model, img_path):
    r = model.predict(source=str(img_path), imgsz=model.predictor.args.imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return 1.0, 0.0
    c = float(r.boxes.conf.max().item())
    return 1.0 - c, c


def measure_infer_time(model, imgs, imgsz, max_n=50):
    sample = imgs[:max_n]
    t0 = time.perf_counter()
    for p in sample:
        _ = model.predict(source=str(p), imgsz=imgsz, verbose=False)
    return (time.perf_counter() - t0) / max(1, len(sample)) * 1000.0


def train_model(weights, data_yaml, out_dir, cfg):
    model = YOLO(weights)
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs_per_cycle,
        batch=cfg.batch,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        verbose=False,
        device=cfg.device,
    )
    train_time = time.perf_counter() - t0
    best = out_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        best = out_dir / "train" / "weights" / "last.pt"
    return str(best), train_time


def eval_model(weights, data_yaml, out_dir, cfg):
    model = YOLO(weights)
    v = model.val(
        data=str(data_yaml),
        split="test",
        project=str(out_dir),
        name="val",
        exist_ok=True,
        verbose=False,
        device=cfg.device,
    )
    return (
        float(getattr(v.box, "map50", np.nan)),
        float(getattr(v.box, "map", np.nan)),
        float(getattr(v.box, "mp", np.nan)),
        float(getattr(v.box, "mr", np.nan)),
    )


def compute_auc_over_budget(df, y_col, x_col="labeled_count", budget_added=400):
    d = df[[x_col, y_col]].dropna().sort_values(x_col)
    if len(d) < 2:
        return float("nan"), float("nan")
    if "cycle" in df.columns and (df["cycle"] == 0).any():
        l0 = float(df.loc[df["cycle"] == 0, x_col].dropna().iloc[0])
    else:
        l0 = float(d[x_col].min())
    x_min, x_max = l0, l0 + float(budget_added)
    x_all = d[x_col].to_numpy(dtype=np.float64)
    y_all = d[y_col].to_numpy(dtype=np.float64)
    interp = lambda xq: float(np.interp(xq, x_all, y_all))
    d_in = d[(d[x_col] >= x_min) & (d[x_col] <= x_max)].copy()
    if len(d_in) == 0 or not np.isclose(float(d_in[x_col].min()), x_min):
        d_in = pd.concat([pd.DataFrame([{x_col: x_min, y_col: interp(x_min)}]), d_in], ignore_index=True)
    if len(d_in) == 0 or not np.isclose(float(d_in[x_col].max()), x_max):
        d_in = pd.concat([d_in, pd.DataFrame([{x_col: x_max, y_col: interp(x_max)}])], ignore_index=True)
    d_in = d_in.sort_values(x_col).drop_duplicates(subset=[x_col], keep="first")
    if len(d_in) < 2:
        return float("nan"), float("nan")
    x = d_in[x_col].to_numpy(dtype=np.float64)
    y = d_in[y_col].to_numpy(dtype=np.float64)
    auc = float(np.trapezoid(y, x))
    span = float(x_max - x_min)
    return auc, float(auc / span) if span > 0 else float("nan")


def setup_reproducibility(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def run_method(method, cfg):
    if method not in {"random", "us_single", "proposed_mc"}:
        raise ValueError

    root = Path(__file__).resolve().parents[1]
    splits = root / "splits"
    raw_labels = root / "data_raw" / "train" / "labels"
    sard_yaml = root / "data_raw" / "data.yaml"

    if not splits.exists():
        raise FileNotFoundError("Pasta splits/ nao encontrada. Rode prepare_splits.py antes.")
    if not sard_yaml.exists():
        raise FileNotFoundError("data_raw/data.yaml nao encontrado.")

    out_dir = Path(cfg.out_root)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir / f"{method}_seed{cfg.seed}"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = out_dir / "data.yaml"
    yolo_data_yaml(splits, data_yaml, sard_yaml)

    setup_reproducibility(cfg)

    pool_dir = splits / "pool" / "images"
    labeled_train_images = splits / "labeled" / "train" / "images"
    labeled_train_labels = splits / "labeled" / "train" / "labels"

    weights, train_time = train_model(cfg.weights0, data_yaml, out_dir / "cycle_0", cfg)
    map50, map5095, prec, rec = eval_model(weights, data_yaml, out_dir / "cycle_0", cfg)

    log = [{
        "cycle": 0,
        "labeled_count": len(list_images(labeled_train_images)),
        "pool_remaining": len(list_images(pool_dir)),
        "added_this_cycle": 0,
        "train_time_s": train_time,
        "selection_time_s": 0.0,
        "infer_ms": np.nan,
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": prec,
        "recall": rec,
        "select_mean_conf": np.nan,
        "select_std_conf": np.nan,
    }]

    for c in range(1, cfg.cycles + 1):
        pool_imgs = list_images(pool_dir)
        if not pool_imgs:
            break

        use_random = method == "random"
        use_mc = method == "proposed_mc"

        sel_model = YOLO(weights)
        sel_model.predictor = None
        _ = sel_model.predict(source=str(pool_imgs[0]), imgsz=cfg.imgsz, verbose=False, device=cfg.device)

        if use_mc:
            sel_model.model.eval()
            set_bn_eval(sel_model.model)
            enable_dropout_only(sel_model.model)

        subset = pool_imgs if cfg.max_pool_eval is None else pool_imgs[: min(len(pool_imgs), cfg.max_pool_eval)]

        t0 = time.perf_counter()
        b = min(cfg.budget_per_cycle, len(pool_imgs))

        if use_random:
            chosen = random.sample(pool_imgs, b)
            select_mean = select_std = 0.0
        else:
            scores, conf_means, conf_stds = [], [], []
            for pth in subset:
                if use_mc:
                    s, m, sd = uncertainty_score_mc(sel_model, pth, cfg.mc_passes)
                    conf_stds.append(sd)
                else:
                    s, m = uncertainty_score_single(sel_model, pth)
                    conf_stds.append(0.0)
                scores.append((s, pth))
                conf_means.append(m)
            scores.sort(key=lambda x: x[0], reverse=True)
            chosen = [p for _, p in scores[:b]]
            idxs = [subset.index(p) for p in chosen if p in subset]
            select_mean = float(np.mean([conf_means[i] for i in idxs])) if idxs else float("nan")
            select_std = float(np.mean([conf_stds[i] for i in idxs])) if idxs else float("nan")

        sel_time = time.perf_counter() - t0

        for p in chosen:
            reveal_label_and_move(p, raw_labels, labeled_train_images, labeled_train_labels)
            try:
                (pool_dir / p.name).unlink()
            except FileNotFoundError:
                pass

        weights, train_time = train_model(weights, data_yaml, out_dir / f"cycle_{c}", cfg)
        map50, map5095, prec, rec = eval_model(weights, data_yaml, out_dir / f"cycle_{c}", cfg)
        infer_ms = measure_infer_time(YOLO(weights), list_images(splits / "test" / "images"), cfg.imgsz, 50)

        log.append({
            "cycle": c,
            "labeled_count": len(list_images(labeled_train_images)),
            "pool_remaining": len(list_images(pool_dir)),
            "added_this_cycle": b,
            "train_time_s": train_time,
            "selection_time_s": sel_time,
            "infer_ms": infer_ms,
            "mAP50": map50,
            "mAP50_95": map5095,
            "precision": prec,
            "recall": rec,
            "select_mean_conf": select_mean,
            "select_std_conf": select_std,
        })

        print(f"[{method}] cycle {c} ok | added={b} | labeled={log[-1]['labeled_count']} | pool={log[-1]['pool_remaining']}")

    df = pd.DataFrame(log)
    df.to_csv(out_dir / "results.csv", index=False)

    auc50, auc50_norm = compute_auc_over_budget(df, "mAP50", "labeled_count", cfg.auc_budget_labels)
    auc5095, auc5095_norm = compute_auc_over_budget(df, "mAP50_95", "labeled_count", cfg.auc_budget_labels)
    l0 = int(df.loc[df["cycle"] == 0, "labeled_count"].iloc[0])

    summary = {
        "method": method,
        "seed": cfg.seed,
        "auc_budget_added": cfg.auc_budget_labels,
        "auc_map50": auc50,
        "auc_map50_norm": auc50_norm,
        "auc_map50_95": auc5095,
        "auc_map50_95_norm": auc5095_norm,
        "final_cycle": int(df.iloc[-1]["cycle"]),
        "final_labeled": int(df.iloc[-1]["labeled_count"]),
        "final_map50": float(df.iloc[-1]["mAP50"]),
        "final_map50_95": float(df.iloc[-1]["mAP50_95"]),
        "final_precision": float(df.iloc[-1]["precision"]),
        "final_recall": float(df.iloc[-1]["recall"]),
        "final_infer_ms": float(df.iloc[-1]["infer_ms"]),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print(f"Saved: {out_dir / 'results.csv'}")
    print(f"Saved: {out_dir / 'summary.csv'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mc_passes", type=int, default=8)
    ap.add_argument("--auc_budget", type=int, default=400)
    ap.add_argument("--weights0", type=str, default="yolov8n.pt")
    ap.add_argument("--max_pool_eval", type=int, default=None)
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no_deterministic", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Cfg(
        seed=a.seed,
        cycles=a.cycles,
        budget_per_cycle=a.budget,
        imgsz=a.imgsz,
        epochs_per_cycle=a.epochs,
        batch=a.batch,
        mc_passes=a.mc_passes,
        auc_budget_labels=a.auc_budget,
        weights0=a.weights0,
        max_pool_eval=a.max_pool_eval,
        out_root=a.out_root,
        device=a.device,
        deterministic=(not a.no_deterministic),
    )

    root = Path(__file__).resolve().parents[1]
    out_base = Path(cfg.out_root)
    if not out_base.is_absolute():
        out_base = root / out_base
    out_base.mkdir(parents=True, exist_ok=True)

    for m in ["random", "us_single", "proposed_mc"]:
        run_method(m, cfg)

    summaries = sorted(out_base.glob("*_seed*/summary.csv"))
    if summaries:
        pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True).to_csv(
            out_base / "summary_all_methods.csv", index=False
        )
        print(f"Saved: {out_base / 'summary_all_methods.csv'}")


if __name__ == "__main__":
    main()
