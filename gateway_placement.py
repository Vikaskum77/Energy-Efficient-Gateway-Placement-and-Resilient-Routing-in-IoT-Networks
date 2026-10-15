"""
Energy-Efficient IoT Gateway Placement using K-Means Clustering
----------------------------------------------------------------
Compares: Random placement  vs  K-Means placement  (and a single central gateway)
Metrics : average distance, energy per round, hotspot (max node energy),
          network lifetime (first node dead / 50% nodes dead)

Run:  pip install numpy matplotlib scikit-learn
      python gateway_placement.py
Outputs: 3 PNG graphs + results.csv + console summary
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------- PARAMETERS -------------------------
SEED = 42
AREA = 200.0             # field size: 200 m x 200 m
N_NODES = 200            # number of IoT sensor nodes
K_LIST = [1, 2, 3, 4, 5, 6, 7, 8]   # gateway counts to test
K_DEMO = 4               # gateway count used for the scatter plot
RANDOM_TRIALS = 30       # random placement is averaged over many trials

# First-order radio energy model (standard in WSN/IoT papers)
E_ELEC = 50e-9           # J/bit  (TX/RX electronics)
E_AMP = 100e-12          # J/bit/m^2 (amplifier)
PKT_BITS = 4000          # packet size in bits
E_INIT = 0.5             # initial battery per node (J)
TX_PROB = 0.5            # probability a node transmits in a round
MAX_ROUNDS = 200000

rng = np.random.default_rng(SEED)


# ------------------------- HELPERS -------------------------
def deploy_nodes(n):
    """Non-uniform deployment: 3 dense hotspots + some uniform background nodes
    (more realistic than purely uniform, and makes K-Means shine)."""
    centers = rng.uniform(0.2 * AREA, 0.8 * AREA, size=(3, 2))
    n_cluster = int(n * 0.7)
    pts = []
    for i in range(n_cluster):
        c = centers[i % 3]
        pts.append(rng.normal(c, 15.0))
    pts = np.array(pts)
    background = rng.uniform(0, AREA, size=(n - n_cluster, 2))
    allp = np.vstack([pts, background])
    return np.clip(allp, 0, AREA)


def assign(nodes, gws):
    """Each node connects to its nearest gateway. Returns distances."""
    d = np.linalg.norm(nodes[:, None, :] - gws[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return d[np.arange(len(nodes)), nearest], nearest


def tx_energy(dist):
    """Energy (J) for one packet transmission over distance dist."""
    return PKT_BITS * (E_ELEC + E_AMP * dist ** 2)


def lifetime(dist):
    """Round-based simulation: returns (first node dead round, 50% dead round)."""
    e_pkt = tx_energy(dist)
    battery = np.full(len(dist), E_INIT)
    fnd, half = None, None
    for r in range(1, MAX_ROUNDS + 1):
        # Dead nodes must never transmit or consume additional battery.
        alive = battery > 0
        sends = alive & (rng.random(len(dist)) < TX_PROB)
        battery[sends] -= e_pkt[sends]
        dead = (battery <= 0).sum()
        if fnd is None and dead >= 1:
            fnd = r
        if half is None and dead >= len(dist) // 2:
            half = r
            break
    return fnd or MAX_ROUNDS, half or MAX_ROUNDS


def evaluate(nodes, gws):
    dist, nearest = assign(nodes, gws)
    e = tx_energy(dist)
    fnd, half = lifetime(dist)
    return {
        "avg_dist": dist.mean(),
        "max_dist": dist.max(),
        "energy_round": e.sum() * TX_PROB, # expected energy/round with TX_PROB
        "full_load_energy_round": e.sum(), # reference if every node transmits
        "max_node_energy": e.max(),       # per-packet node energy (hotspot)
        "fnd": fnd,
        "half": half,
    }


def kmeans_gateways(nodes, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(nodes)
    return km.cluster_centers_


def random_gateways(k):
    return rng.uniform(0, AREA, size=(k, 2))


# ------------------------- MAIN EXPERIMENT -------------------------
nodes = deploy_nodes(N_NODES)
results = {"kmeans": [], "random": []}

for k in K_LIST:
    results["kmeans"].append(evaluate(nodes, kmeans_gateways(nodes, k)))
    trials = [evaluate(nodes, random_gateways(k)) for _ in range(RANDOM_TRIALS)]
    results["random"].append({m: np.mean([t[m] for t in trials]) for m in trials[0]})

# single central gateway baseline
center = np.array([[AREA / 2, AREA / 2]])
central = evaluate(nodes, center)

# ------------------------- CONSOLE + CSV -------------------------
print(f"{'k':>2} | {'Method':<8} | {'AvgDist(m)':>10} | {'Energy/round(mJ)':>16} | {'FND':>6} | {'50% dead':>8}")
print("-" * 66)
rows = ["k,method,avg_dist_m,max_dist_m,expected_energy_per_round_J,full_load_energy_per_round_J,max_node_packet_energy_J,FND_round,half_dead_round"]
for i, k in enumerate(K_LIST):
    for m in ("random", "kmeans"):
        r = results[m][i]
        print(f"{k:>2} | {m:<8} | {r['avg_dist']:>10.2f} | {r['energy_round']*1e3:>16.3f} | {r['fnd']:>6.0f} | {r['half']:>8.0f}")
        rows.append(f"{k},{m},{r['avg_dist']:.3f},{r['max_dist']:.3f},{r['energy_round']:.6f},{r['full_load_energy_round']:.6f},{r['max_node_energy']:.8f},{r['fnd']:.0f},{r['half']:.0f}")
print(f"\nCentral single gateway: avg_dist={central['avg_dist']:.2f} m, FND={central['fnd']}")
open("results.csv", "w").write("\n".join(rows))

# improvement summary for demo k
i = K_LIST.index(K_DEMO)
imp = (1 - results["kmeans"][i]["energy_round"] / results["random"][i]["energy_round"]) * 100
print(f"\nAt k={K_DEMO}: K-Means saves {imp:.1f}% expected energy/round vs random placement.")

# ------------------------- PLOTS -------------------------
# 1) Placement scatter: random vs k-means
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, title, gws in [
    (axes[0], f"Random placement (k={K_DEMO})", random_gateways(K_DEMO)),
    (axes[1], f"K-Means placement (k={K_DEMO})", kmeans_gateways(nodes, K_DEMO)),
]:
    dist, nearest = assign(nodes, gws)
    ax.scatter(nodes[:, 0], nodes[:, 1], c=nearest, cmap="tab10", s=18, alpha=0.8)
    for n_i, g in zip(range(len(nodes)), nearest):
        pass
    ax.scatter(gws[:, 0], gws[:, 1], marker="*", s=350, c="red", edgecolors="black", label="Gateway")
    ax.set_title(f"{title}\navg distance = {dist.mean():.1f} m")
    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig("1_placement_comparison.png", dpi=150); plt.close()

# 2) Energy per round vs number of gateways
plt.figure(figsize=(7, 5))
plt.plot(K_LIST, [r["energy_round"] * 1e3 for r in results["random"]], "o--", label="Random")
plt.plot(K_LIST, [r["energy_round"] * 1e3 for r in results["kmeans"]], "s-", label="K-Means")
plt.axhline(central["energy_round"] * 1e3, color="gray", ls=":", label="Single central gateway")
plt.xlabel("Number of gateways (k)"); plt.ylabel("Energy per round (mJ)")
plt.title("Energy consumption vs number of gateways"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("2_energy_vs_k.png", dpi=150); plt.close()

# 3) Network lifetime vs number of gateways
plt.figure(figsize=(7, 5))
plt.plot(K_LIST, [r["fnd"] for r in results["random"]], "o--", label="Random (first node dead)")
plt.plot(K_LIST, [r["fnd"] for r in results["kmeans"]], "s-", label="K-Means (first node dead)")
plt.xlabel("Number of gateways (k)"); plt.ylabel("Rounds until first node dies")
plt.title("Network lifetime vs number of gateways"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("3_lifetime_vs_k.png", dpi=150); plt.close()

print("\nSaved: 1_placement_comparison.png, 2_energy_vs_k.png, 3_lifetime_vs_k.png, results.csv")
