import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ==========================
# CONFIG
# ==========================
BASE_PATH = Path("/home/pdi-b06/Marigold")

datasets = {
    "UIEB": {
        "rgb": BASE_PATH / "input/uieb",
        "models": {
            "Small": BASE_PATH / "output/marigold_small/depth_npy",
            "Base" :  BASE_PATH / "output/marigold_base/depth_npy",
            "Large": BASE_PATH / "output/marigold_large/depth_npy",
        }
    },
    "SUIM": {
        "rgb": BASE_PATH / "input/suim",
        "models": {
            "Small": BASE_PATH / "output/mg-suim_small/depth_npy",
            "Base" :  BASE_PATH / "output/mg-suim_base/depth_npy",
            "Large": BASE_PATH / "output/mg-suim_large/depth_npy",
        }
    },
    "USIS10K": {
        "rgb": BASE_PATH / "input/usis10k",
        "models": {
            "Small": BASE_PATH / "output/mg-usis10k_small/depth_npy",
            "Base" :  BASE_PATH / "output/mg-usis10k_base/depth_npy",
            "Large": BASE_PATH / "output/mg-usis10k_large/depth_npy",
        }
    }
}

OUT_PATH = BASE_PATH / "output/metrics_results/no_ref_metrics.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==========================
# MÉTRICAS
# ==========================

def edge_alignment_score(rgb, depth):
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges_rgb = cv2.Canny(rgb_gray, 50, 150)

    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    if grad_mag.max() < 1e-6:
        return 0.0

    grad_mag = grad_mag / grad_mag.max()
    edges_depth = (grad_mag > 0.1)

    intersection = np.logical_and(edges_rgb > 0, edges_depth).sum()
    union = np.logical_or(edges_rgb > 0, edges_depth).sum()

    return intersection / (union + 1e-8)


def depth_smoothness(depth):
    dx = np.abs(np.diff(depth, axis=1))
    dy = np.abs(np.diff(depth, axis=0))
    return float((dx.mean() + dy.mean()) / 2.0)


def edge_aware_smoothness(rgb, depth):
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    dx_depth = np.abs(np.diff(depth, axis=1))
    dy_depth = np.abs(np.diff(depth, axis=0))

    dx_img = np.abs(np.diff(rgb_gray, axis=1))
    dy_img = np.abs(np.diff(rgb_gray, axis=0))

    weight_x = np.exp(-dx_img)
    weight_y = np.exp(-dy_img)

    smooth_x = dx_depth * weight_x
    smooth_y = dy_depth * weight_y

    return float((smooth_x.mean() + smooth_y.mean()) / 2.0)


# ==========================
# LOOP PRINCIPAL
# ==========================

results = []

for ds_name, ds in datasets.items():

    rgb_files = sorted(list(ds["rgb"].glob("*.jpg")) + list(ds["rgb"].glob("*.png")))

    print(f"\n📂 Dataset: {ds_name} | {len(rgb_files)} imagens")

    for model_name, depth_dir in ds["models"].items():

        metrics_stack = []

        for idx, rgb_path in enumerate(rgb_files):
            base_name = rgb_path.stem
            depth_path = depth_dir / f"{base_name}_depth.npy"

            if not depth_path.exists():
                continue

            # Load RGB
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Load Depth
            depth = np.load(depth_path).astype(np.float32)

            # Resize se necessário
            if rgb.shape[:2] != depth.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

            # Métricas
            e = edge_alignment_score(rgb, depth)
            s = depth_smoothness(depth)
            eas = edge_aware_smoothness(rgb, depth)

            metrics_stack.append([e, s, eas])

            if idx % 100 == 0:
                print(f"{model_name}: {idx}/{len(rgb_files)}")

        if metrics_stack:
            avg = np.mean(metrics_stack, axis=0)

            results.append([
                ds_name,
                model_name,
                avg[0],
                avg[1],
                avg[2]
            ])

# ==========================
# DATAFRAME
# ==========================

df = pd.DataFrame(results, columns=[
    "Dataset",
    "Model",
    "Edge Align (↑)",
    "Smoothness (↓)",
    "EdgeAwareSmooth (↓)"
])

# ==========================
# OUTPUT
# ==========================

print("\n" + "="*80)
print("📊 MÉTRICAS SEM GROUND TRUTH")
print("="*80)
print(df.to_string(index=False))
print("="*80)

df.to_csv(OUT_PATH, index=False)

print(f"\n✅ CSV salvo em: {OUT_PATH}")