import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path):
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS])


def copy_pair(img_path: Path, src_labels: Path, dst_images: Path, dst_labels: Path):
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst_images / img_path.name)
    lbl = src_labels / (img_path.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, dst_labels / lbl.name)
    else:
        (dst_labels / (img_path.stem + ".txt")).write_text("")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l0", type=int, default=200)
    ap.add_argument("--pool_cap", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="splits")
    return ap.parse_args()


def main():
    a = parse_args()
    root = Path(__file__).resolve().parents[1]
    sard = root / "data_raw"
    if not sard.exists():
        raise FileNotFoundError("Dataset data_raw/ nao encontrado")

    train_images = sard / "train" / "images"
    train_labels = sard / "train" / "labels"
    valid_images = sard / "valid" / "images"
    valid_labels = sard / "valid" / "labels"
    test_images = sard / "test" / "images"
    test_labels = sard / "test" / "labels"

    for p in [train_images, train_labels, valid_images, valid_labels, test_images, test_labels]:
        if not p.exists():
            raise FileNotFoundError(f"Faltando: {p}")

    out = Path(a.out_dir)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)

    for img in list_images(test_images):
        copy_pair(img, test_labels, out / "test" / "images", out / "test" / "labels")

    for img in list_images(valid_images):
        copy_pair(img, valid_labels, out / "labeled" / "val" / "images", out / "labeled" / "val" / "labels")

    pool_imgs = list_images(train_images)
    random.shuffle(pool_imgs)

    if a.pool_cap is not None:
        if a.pool_cap < 1:
            raise ValueError("pool_cap deve ser >= 1")
        pool_imgs = pool_imgs[: min(len(pool_imgs), a.pool_cap)]

    if a.l0 >= len(pool_imgs):
        raise ValueError(f"l0 muito grande: l0={a.l0} >= {len(pool_imgs)} (pool disponivel)")

    l0_imgs = pool_imgs[: a.l0]
    pool_rest = pool_imgs[a.l0 :]

    for img in l0_imgs:
        copy_pair(img, train_labels, out / "labeled" / "train" / "images", out / "labeled" / "train" / "labels")

    (out / "pool" / "images").mkdir(parents=True, exist_ok=True)
    for img in pool_rest:
        shutil.copy2(img, out / "pool" / "images" / img.name)

    print("Split pronto")
    print(f"Train total: {len(list_images(train_images))}")
    print(f"Valid fixo: {len(list_images(valid_images))}")
    print(f"Test fixo: {len(list_images(test_images))}")
    print(f"L0 treino: {len(l0_imgs)}")
    print(f"Pool: {len(pool_rest)}")
    print(f"Out: {out}")


if __name__ == "__main__":
    main()
