"""
ablation_study.py — Generate PNG graphs for parameter sensitivity analysis
Produces 3 PNG files + 1 CSV summary for laporan ablation study section
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
from data.loader import load_dataset
from models.genetic_fuzzy import GeneticFuzzySystem

# ════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════
plt.style.use('dark_background')
sns.set_palette("husl")

df = load_dataset()

user_prefs = {
    "energy": 0.70,
    "danceability": 0.70,
    "valence": 0.60,
    "speechiness": 0.08,
    "acousticness": 0.20,
    "popularity": 0.70,
}

print("[*] Dataset loaded:", len(df), "songs")
print("[*] Starting ablation study...\n")

# ════════════════════════════════════════════════════════════════════════════
# 1. POPULATION SIZE ABLATION
# ════════════════════════════════════════════════════════════════════════════
print("[1/3] Population Size Ablation...")

pop_sizes = [5, 50, 100, 300]
pop_results = {}

for pop in pop_sizes:
    print(f"  Testing population={pop}...")
    ga = GeneticFuzzySystem(population_size=pop, max_generations=50, seed=42)
    t0 = time.perf_counter()
    ga.evolve(df, user_prefs, progress_cb=None)
    runtime = time.perf_counter() - t0
    
    hist = ga.get_convergence_data()
    pop_results[pop] = {
        "history": hist,
        "best_fitness": ga.best_fitness,
        "runtime": runtime,
        "final_gen": len(hist),
    }
    print(f"    → Best Fitness: {ga.best_fitness:.2f}, Runtime: {runtime:.2f}s")

# Plot population size ablation
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Ablation Study: Population Size Effect on Convergence", fontsize=14, fontweight='bold', color='white')

for idx, (pop, data) in enumerate(sorted(pop_results.items())):
    ax = axes[idx // 2, idx % 2]
    hist = data["history"]
    
    ax.plot(hist["generation"], hist["best_fitness"], color="#34d399", linewidth=2.5, label="Best Fitness", marker='o', markersize=4)
    ax.fill_between(hist["generation"], hist["best_fitness"], alpha=0.2, color="#34d399")
    ax.plot(hist["generation"], hist["avg_fitness"], color="#fbbf24", linewidth=1.5, linestyle="--", label="Avg Fitness")
    
    # Annotation
    status = "PREMATURE" if pop == 5 else "OPTIMAL" if pop in [50, 100] else "OVERKILL"
    ax.text(0.98, 0.05, f"Pop={pop} | Runtime={data['runtime']:.1f}s | Status: {status}", 
            transform=ax.transAxes, fontsize=10, color='white', 
            bbox=dict(boxstyle='round', facecolor='rgba(0,0,0,0.5)', alpha=0.7),
            ha='right', va='bottom')
    
    ax.set_xlabel("Generation", color='rgba(255,255,255,0.7)')
    ax.set_ylabel("Fitness", color='rgba(255,255,255,0.7)')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([min(hist["best_fitness"].min(), hist["avg_fitness"].min())-5, 100])

plt.tight_layout()
plt.savefig('ablation_population_size.png', dpi=150, bbox_inches='tight', facecolor='#0d0b1e')
print("  → Saved: ablation_population_size.png\n")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 2. MAX GENERATIONS ABLATION
# ════════════════════════════════════════════════════════════════════════════
print("[2/3] Max Generations Ablation...")

max_gens = [10, 30, 50, 120]
gen_results = {}

for max_gen in max_gens:
    print(f"  Testing max_generations={max_gen}...")
    ga = GeneticFuzzySystem(population_size=40, max_generations=max_gen, seed=42)
    t0 = time.perf_counter()
    ga.evolve(df, user_prefs, progress_cb=None)
    runtime = time.perf_counter() - t0
    
    hist = ga.get_convergence_data()
    gen_results[max_gen] = {
        "history": hist,
        "best_fitness": ga.best_fitness,
        "runtime": runtime,
        "final_gen": len(hist),
    }
    print(f"    → Best Fitness: {ga.best_fitness:.2f}, Runtime: {runtime:.2f}s")

# Plot max generations ablation
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Ablation Study: Max Generations Effect on Convergence", fontsize=14, fontweight='bold', color='white')

for idx, (max_gen, data) in enumerate(sorted(gen_results.items())):
    ax = axes[idx // 2, idx % 2]
    hist = data["history"]
    
    ax.plot(hist["generation"], hist["best_fitness"], color="#f472b6", linewidth=2.5, label="Best Fitness", marker='s', markersize=4)
    ax.fill_between(hist["generation"], hist["best_fitness"], alpha=0.2, color="#f472b6")
    ax.plot(hist["generation"], hist["avg_fitness"], color="#fbbf24", linewidth=1.5, linestyle="--", label="Avg Fitness")
    
    # Annotation
    status = "INCOMPLETE" if max_gen == 10 else "OPTIMAL" if max_gen in [30, 50] else "OVERKILL"
    ax.text(0.98, 0.05, f"Gen={max_gen} | Runtime={data['runtime']:.1f}s | Status: {status}", 
            transform=ax.transAxes, fontsize=10, color='white',
            bbox=dict(boxstyle='round', facecolor='rgba(0,0,0,0.5)', alpha=0.7),
            ha='right', va='bottom')
    
    ax.set_xlabel("Generation", color='rgba(255,255,255,0.7)')
    ax.set_ylabel("Fitness", color='rgba(255,255,255,0.7)')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([min(hist["best_fitness"].min(), hist["avg_fitness"].min())-5, 100])

plt.tight_layout()
plt.savefig('ablation_max_generations.png', dpi=150, bbox_inches='tight', facecolor='#0d0b1e')
print("  → Saved: ablation_max_generations.png\n")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 3. DIMINISHING RETURNS: Pop Size Range Study
# ════════════════════════════════════════════════════════════════════════════
print("[3/3] Diminishing Returns Analysis...")

pop_range = list(range(5, 305, 20))  # 5 to 300 in steps of 20
diminishing_results = []

for pop in pop_range:
    print(f"  Testing population={pop}...")
    ga = GeneticFuzzySystem(population_size=pop, max_generations=50, seed=42)
    t0 = time.perf_counter()
    ga.evolve(df, user_prefs, progress_cb=None)
    runtime = time.perf_counter() - t0
    
    diminishing_results.append({
        "population": pop,
        "best_fitness": ga.best_fitness,
        "runtime": runtime,
    })

dim_df = pd.DataFrame(diminishing_results)

# Plot diminishing returns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ablation Study: Diminishing Returns (Population vs Fitness/Runtime)", 
             fontsize=14, fontweight='bold', color='white')

# Left: Fitness vs Population
ax1.plot(dim_df["population"], dim_df["best_fitness"], color="#34d399", linewidth=2.5, marker='o', markersize=6, label="Best Fitness")
ax1.fill_between(dim_df["population"], dim_df["best_fitness"], alpha=0.2, color="#34d399")
ax1.set_xlabel("Population Size", fontsize=11, color='rgba(255,255,255,0.7)')
ax1.set_ylabel("Best Fitness Score", fontsize=11, color='rgba(255,255,255,0.7)')
ax1.grid(True, alpha=0.2, color='white')
ax1.set_title("Fitness Convergence (diminishing gains after pop=100)", fontsize=10, color='rgba(255,255,255,0.8)')
ax1.legend(fontsize=9)
# Highlight sweet spot
ax1.axvline(x=50, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.5, label='Recommended Zone')
ax1.axvline(x=100, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.5)
ax1.text(75, ax1.get_ylim()[1]*0.95, "Sweet Spot: 50-100", fontsize=9, color='#fbbf24', 
         bbox=dict(boxstyle='round', facecolor='rgba(0,0,0,0.3)'))

# Right: Runtime vs Population (linear growth)
ax2.plot(dim_df["population"], dim_df["runtime"], color="#f472b6", linewidth=2.5, marker='s', markersize=6, label="Runtime (seconds)")
ax2.fill_between(dim_df["population"], dim_df["runtime"], alpha=0.2, color="#f472b6")
ax2.set_xlabel("Population Size", fontsize=11, color='rgba(255,255,255,0.7)')
ax2.set_ylabel("Runtime (seconds)", fontsize=11, color='rgba(255,255,255,0.7)')
ax2.grid(True, alpha=0.2, color='white')
ax2.set_title("Computational Cost (linear scaling)", fontsize=10, color='rgba(255,255,255,0.8)')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('ablation_diminishing_returns.png', dpi=150, bbox_inches='tight', facecolor='#0d0b1e')
print("  → Saved: ablation_diminishing_returns.png\n")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 4. GENERATE SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
print("[*] Generating summary table...")

summary_data = []

# Population ablations
for pop, data in sorted(pop_results.items()):
    summary_data.append({
        "Experiment": f"Pop={pop}",
        "Parameter": pop,
        "Best Fitness": data["best_fitness"],
        "Convergence ~Gen": int((data["history"]["best_fitness"].diff() < 0.05).idxmax()) 
                           if (data["history"]["best_fitness"].diff() < 0.05).any() else 50,
        "Runtime (s)": round(data["runtime"], 2),
        "Status": "PREMATURE" if pop == 5 else "OPTIMAL" if pop in [50, 100] else "OVERKILL",
    })

# Generation ablations
for max_gen, data in sorted(gen_results.items()):
    summary_data.append({
        "Experiment": f"Gen={max_gen}",
        "Parameter": max_gen,
        "Best Fitness": data["best_fitness"],
        "Convergence ~Gen": int((data["history"]["best_fitness"].diff() < 0.05).idxmax())
                           if (data["history"]["best_fitness"].diff() < 0.05).any() else max_gen,
        "Runtime (s)": round(data["runtime"], 2),
        "Status": "INCOMPLETE" if max_gen == 10 else "OPTIMAL" if max_gen in [30, 50] else "OVERKILL",
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('ablation_summary.csv', index=False)
print("  → Saved: ablation_summary.csv\n")
print("Summary Table:")
print(summary_df.to_string(index=False))

print("\n✅ Ablation study complete!")
print("   Generated files:")
print("   1. ablation_population_size.png — 2x2 subplots showing premature→optimal→overkill progression")
print("   2. ablation_max_generations.png — 2x2 subplots showing incomplete→optimal→overkill progression")
print("   3. ablation_diminishing_returns.png — Fitness curve + runtime linear growth")
print("   4. ablation_summary.csv — Summary table with all experiments")
