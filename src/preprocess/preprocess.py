import random
import shutil
from pathlib import Path


def split_dataset(src="data/raw", dst="data/processed", train=0.7, val=0.15, seed=42):
    random.seed(seed)
    src_path = Path(src)
    dst_path = Path(dst)

    for class_dir in sorted(src_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(train * n)
        n_val = int(val * n)

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for split, imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            out_dir = dst_path / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in imgs:
                shutil.copy(img_path, out_dir / img_path.name)


if __name__ == "__main__":
    split_dataset()
