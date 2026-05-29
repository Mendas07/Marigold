import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para profundidades (em tons de cinza!)
pred_dir = "../../output/test-img/depth_colored"

# Pasta de saída para salvar resultados
os.makedirs("output/metrics_results", exist_ok=True)

results = []
all_depths_combined = []

for f in os.listdir(pred_dir):
    if f.endswith(".png"):
        path = os.path.join(pred_dir, f)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            continue

        # Se for imagem colorida (colormap), converte para cinza
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        depth = depth.astype(np.float32)

        mean_val = float(np.mean(depth))
        min_val = float(np.min(depth))
        max_val = float(np.max(depth))
        std_val = float(np.std(depth))

        # Define "near" como 0–25% dos valores possíveis, "far" como 75–100%
        threshold_near = max_val * 0.25
        threshold_far = max_val * 0.75

        total_pixels = depth.size
        near_percent = float(np.sum(depth <= threshold_near) / total_pixels * 100)
        far_percent = float(np.sum(depth >= threshold_far) / total_pixels * 100)

        ratio_near_far = near_percent / far_percent if far_percent > 0 else np.inf

        results.append({
            "image": f,
            "mean_depth": mean_val,
            "min_depth": min_val,
            "max_depth": max_val,
            "std_depth": std_val,
            "near_percent": near_percent,
            "far_percent": far_percent,
            "ratio_near_far": ratio_near_far
        })

        all_depths_combined.append(depth.flatten())

# Evita erro caso não encontre imagens
if not results:
    print(f"Nenhuma imagem encontrada em {pred_dir}")
    exit()

# Salva métricas detalhadas em CSV
df = pd.DataFrame(results)
csv_path = "output/metrics_results/depth_detailed_statistics.csv"
df.to_csv(csv_path, index=False)
print(f"📊 Estatísticas detalhadas salvas em {csv_path}")

# Calcula médias globais
summary = df.mean(numeric_only=True)
summary_csv_path = "output/metrics_results/depth_detailed_summary.csv"
summary.to_csv(summary_csv_path, header=["value"])
print(f"📊 Resumo detalhado salvo em {summary_csv_path}")
print("\nMédias gerais de todas as imagens:")
print(summary)

# Cria histograma combinado
all_depths_combined = np.concatenate(all_depths_combined)
plt.hist(all_depths_combined, bins=50, color='blue', alpha=0.7)
plt.title("Histograma combinado das profundidades")
plt.xlabel("Valor de profundidade (0-255)")
plt.ylabel("Frequência")
plt.savefig("output/metrics_results/combined_hist.png")
plt.close()
print("📈 Histograma combinado salvo em output/metrics_results/combined_hist.png")
