import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# CONFIG
# ==========================
BASE_PATH = Path("/home/pdi-b06/Marigold")
CSV_PATH = BASE_PATH / "output/metrics_results/no_ref_metrics.csv"

OUT_DIR = BASE_PATH / "output/metrics_results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_PATH = OUT_DIR / "benchmark_table.png"

# ==========================
# LOAD CSV
# ==========================
df = pd.read_csv(CSV_PATH)

# Limpa texto
df["Model"] = df["Model"].str.replace("⭐", "", regex=False)
df["Model"] = df["Model"].str.replace("<-- BEST", "", regex=False)
df["Model"] = df["Model"].str.strip()

# ==========================
# MÉTRICAS
# ==========================
metrics = [
    "Edge Align (↑)",
    "Smoothness (↓)",
    "EdgeAwareSmooth (↓)"
]

# ==========================
# ==========================
# 🔹 1. TABELA (PNG)
# ==========================
# ==========================

df_table = df.copy()

# Formatação numérica
for col in df_table.columns[2:]:
    df_table[col] = df_table[col].astype(float).map(lambda x: f"{x:.6f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

table = ax.table(
    cellText=df_table.values,
    colLabels=df_table.columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Estilo
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#1f2c3a")
        cell.get_text().set_color("white")
    else:
        cell.set_facecolor("#f2f2f2")

# Linha separadora por dataset
for i in range(1, len(df_table)):
    if df_table.iloc[i]["Dataset"] != df_table.iloc[i-1]["Dataset"]:
        for col in range(len(df_table.columns)):
            table[(i, col)].set_edgecolor("black")
            table[(i, col)].set_linewidth(2)

plt.title("Benchmarking de Desempenho: Marigold (No-Reference Metrics)",
          fontsize=18, weight='bold', pad=20)

plt.savefig(TABLE_PATH, bbox_inches='tight', dpi=300)
plt.close()

print(f"✔️ Tabela salva: {TABLE_PATH}")

# ==========================
# ==========================
# 🔹 2. PLOTS COMPARATIVOS
# ==========================
# ==========================

for metric in metrics:

    plt.figure()

    for dataset in df["Dataset"].unique():
        df_ds = df[df["Dataset"] == dataset]
        df_ds = df_ds.sort_values("Model")

        plt.plot(
            df_ds["Model"],
            df_ds[metric],
            marker='o',
            label=dataset
        )

    plt.title(f"Comparativo - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    save_path = OUT_DIR / f"comparativo_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"✔️ Plot salvo: {save_path}")

print("\n✅ Tudo gerado com sucesso!")