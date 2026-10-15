"""
Extended IoT Gateway Placement  (builds on gateway_placement.py)
 Part A: Optimal number of gateways  (elbow + silhouette + total cost)
 Part B: Fault tolerance             (backup gateway, worst-case gateway failure)
 Part C: Multi-hop routing           (direct vs multi-hop vs residual-energy-aware multi-hop)
Run: pip install numpy matplotlib scikit-learn ; python gateway_extended.py
"""
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED, AREA, N = 42, 200.0, 200
E_ELEC, E_AMP, PKT, E0, TXP = 50e-9, 100e-12, 4000, 0.5, 0.5
RX = PKT * E_ELEC                      # energy to receive one packet
R_COMM = 80.0                          # max direct node->gateway range (m)
R_HOP = 40.0                           # max node->node relay range (m)
GW_COST, BATT_COST, HORIZON = 200.0, 5.0, 5000   # cost model for Part A
K_MAX, K_LIST = 10, list(range(1, 9))
RECOMP, ALPHA, MAX_ROUNDS = 50, 1.0, 8000
K_MH = 2                               # sparse gateways: where multi-hop really matters
rng = np.random.default_rng(SEED)


def etx(d):
    return PKT * (E_ELEC + E_AMP * np.square(d))


def dmat(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def deploy(n):
    c = rng.uniform(0.2 * AREA, 0.8 * AREA, (3, 2))
    nc = int(n * 0.7)
    p = np.array([rng.normal(c[i % 3], 15) for i in range(nc)])
    return np.clip(np.vstack([p, rng.uniform(0, AREA, (n - nc, 2))]), 0, AREA)


def kmeans(nodes, k):
    m = KMeans(k, n_init=10, random_state=SEED).fit(nodes)
    return m.cluster_centers_, m.inertia_, m.labels_


nodes = deploy(N)

# ======================= PART A: optimal k =======================
ks = list(range(1, K_MAX + 1))
inertia, sil, gcost, bcost = [], [], [], []
for k in ks:
    g, inn, lab = kmeans(nodes, k)
    e = etx(dmat(nodes, g).min(1)).sum()
    inertia.append(inn)
    sil.append(silhouette_score(nodes, lab) if k > 1 else np.nan)
    gcost.append(k * GW_COST)
    bcost.append(e * TXP * HORIZON / E0 * BATT_COST)   # battery packs used x price
total = np.array(gcost) + np.array(bcost)
x = np.linspace(0, 1, K_MAX)
y = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
k_elbow = ks[int(np.argmax((y[0] + (y[-1] - y[0]) * x) - y))]
k_sil = ks[int(np.nanargmax(sil))]
k_cost = ks[int(np.argmin(total))]
print(f"[A] elbow k={k_elbow} | best silhouette k={k_sil} | LOWEST TOTAL COST k={k_cost}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].plot(ks, inertia, "o-"); ax[0].axvline(k_elbow, color="r", ls=":")
ax[0].set(title="Elbow method", xlabel="k", ylabel="K-Means inertia")
ax[1].plot(ks, sil, "s-"); ax[1].axvline(k_sil, color="r", ls=":")
ax[1].set(title="Silhouette score", xlabel="k")
ax[2].bar(ks, gcost, label="Gateway cost"); ax[2].bar(ks, bcost, bottom=gcost, label="Battery replacement cost")
ax[2].axvline(k_cost, color="r", ls=":"); ax[2].legend()
ax[2].set(title=f"Total cost (min at k={k_cost})", xlabel="k", ylabel="Cost units")
plt.tight_layout(); plt.savefig("4_optimal_k.png", dpi=150); plt.close()
K_OPT = k_cost

# ======================= PART B: fault tolerance =======================
def delivery(g, failed, backup):
    """Fraction of nodes that can still deliver data after gateways in `failed` die."""
    D = dmat(nodes, g)
    alive = np.ones(len(g), bool); alive[failed] = False
    ok = (D <= R_COMM) & alive[None, :]
    if backup:
        return ok.any(1).mean()                       # any working gateway in range
    prim = D.argmin(1)                                # only the primary gateway is known
    return ok[np.arange(len(nodes)), prim].mean()


def worst_case(g, backup):
    return min(delivery(g, [j], backup) for j in range(len(g)))


GRID = np.array([[a, b] for a in np.linspace(5, AREA - 5, 20) for b in np.linspace(5, AREA - 5, 20)])


def add_spare(g):
    """Place ONE extra gateway at the grid point that gives the best worst-case delivery."""
    best = max(GRID, key=lambda c: worst_case(np.vstack([g, c]), True))
    return np.vstack([g, best])


res = {"Random + backup": [], "K-Means, no backup": [], "K-Means + backup": [], "K-Means + backup + 1 smart spare gateway": []}
for k in K_LIST:
    gk = kmeans(nodes, k)[0]
    res["K-Means, no backup"].append(worst_case(gk, False))
    res["K-Means + backup"].append(worst_case(gk, True))
    res["K-Means + backup + 1 smart spare gateway"].append(worst_case(add_spare(gk), True))
    res["Random + backup"].append(np.mean([worst_case(rng.uniform(0, AREA, (k, 2)), True) for _ in range(30)]))
plt.figure(figsize=(8, 5))
for name, v in res.items():
    plt.plot(K_LIST, np.array(v) * 100, "o-", label=name)
plt.xlabel("Number of gateways (k)"); plt.ylabel("Delivery ratio after worst single gateway failure (%)")
plt.title("Fault tolerance"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig("5_fault_tolerance.png", dpi=150); plt.close()
i = K_LIST.index(min(K_OPT, K_LIST[-1]))
print(f"[B] k={K_LIST[i]}: worst-case delivery  no-backup={res['K-Means, no backup'][i]*100:.1f}%  "
      f"backup={res['K-Means + backup'][i]*100:.1f}%  backup+spare={res['K-Means + backup + 1 smart spare gateway'][i]*100:.1f}%")

# ======================= PART C: multi-hop routing =======================
GW = kmeans(nodes, K_MH)[0]
D, DG = dmat(nodes, nodes), dmat(nodes, GW)


def build_routes(alive, bat, mode):
    """nxt[i] = -1-j  -> send straight to gateway j ;  nxt[i] = m >= 0 -> relay via node m ; -99 unreachable"""
    w = (E0 / np.maximum(bat, 1e-3)) ** ALPHA if mode == "aware" else np.ones(N)
    cost, nxt, heap = np.full(N, np.inf), np.full(N, -99), []
    for i in np.flatnonzero(alive):
        j = int(DG[i].argmin())
        if DG[i, j] <= R_COMM:
            cost[i], nxt[i] = etx(DG[i, j]) * w[i], -1 - j
            heapq.heappush(heap, (cost[i], i))
    if mode == "direct":
        return nxt
    while heap:                                       # Dijkstra: cheapest energy path to a gateway
        c, i = heapq.heappop(heap)
        if c > cost[i]:
            continue
        for j in np.flatnonzero((D[i] <= R_HOP) & alive):
            nc = etx(D[i, j]) * w[j] + RX * w[i] + c
            if nc < cost[j]:
                cost[j], nxt[j] = nc, i
                heapq.heappush(heap, (nc, j))
    return nxt


def simulate(mode):
    # Each mode gets the same random stream for repeatability; modes remain
    # stochastic simulations, so publication-grade comparisons should use
    # multiple paired deployment/traffic seeds and report dispersion.
    local_rng = np.random.default_rng(SEED + {"direct": 101, "multihop": 101, "aware": 101}[mode])
    bat, alive = np.full(N, E0), np.ones(N, bool)
    hist, fnd, half, delivered, last, dirty = [], None, None, 0, 0, False
    for r in range(MAX_ROUNDS):
        if r == 0 or r % RECOMP == 0 or (dirty and r - last >= 10):   # refresh routes
            nxt = build_routes(alive, bat, mode); last, dirty = r, False
        if r % 10 == 0:
            hist.append(alive.mean())
        # Nodes are considered alive at the start of the round; depletion is
        # applied at the round boundary (a synchronous-round model).
        for i in np.flatnonzero(alive & (local_rng.random(N) < TXP)):
            cur = i
            for _ in range(30):
                nx = nxt[cur]
                if nx == -99 or (nx >= 0 and not alive[nx]):
                    break
                if nx < 0:
                    bat[cur] -= etx(DG[cur, -1 - nx]); delivered += 1; break
                bat[cur] -= etx(D[cur, nx]); bat[nx] -= RX; cur = nx
        dead = alive & (bat <= 0)
        if dead.any():
            alive &= ~dead; dirty = True
            if fnd is None: fnd = r + 1
            if half is None and (~alive).sum() >= N // 2:
                half = r + 1; break
    return {"hist": hist, "fnd": fnd or MAX_ROUNDS, "half": half or MAX_ROUNDS, "delivered": delivered}


modes = {"direct": "Direct (single hop)", "multihop": "Multi-hop (min energy)", "aware": "Multi-hop (residual-energy aware)"}
out = {m: simulate(m) for m in modes}
print(f"[C] k={K_MH}  (direct link range {R_COMM:.0f} m, relay range {R_HOP:.0f} m)")
for m in modes:
    print(f"    {modes[m]:<34} nodes able to reach a gateway: {(build_routes(np.ones(N, bool), np.full(N, E0), m) != -99).mean()*100:.1f}%")
for m, n in modes.items():
    o = out[m]
    print(f"    {n:<34} first node dead: {o['fnd']:>5} | 50% dead: {o['half']:>5} | packets delivered: {o['delivered']}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for m, n in modes.items():
    ax[0].plot(np.arange(len(out[m]["hist"])) * 10, np.array(out[m]["hist"]) * 100, label=n)
ax[0].set(xlabel="Round", ylabel="Alive nodes (%)", title=f"Network lifetime (k={K_MH})"); ax[0].legend(); ax[0].grid(alpha=.3)
xx = np.arange(3)
ax[1].bar(xx - .2, [out[m]["fnd"] for m in modes], .4, label="First node dead")
ax[1].bar(xx + .2, [out[m]["half"] for m in modes], .4, label="50% nodes dead")
ax[1].set_xticks(xx); ax[1].set_xticklabels(["Direct", "Multi-hop", "Aware"]); ax[1].legend(); ax[1].set_ylabel("Rounds")
plt.tight_layout(); plt.savefig("6_multihop_lifetime.png", dpi=150); plt.close()

rows = ["part,item,value"]
rows += [f"A,k={k},total_cost={t:.1f}" for k, t in zip(ks, total)]
rows += [f"B,{n} k={k},{v*100:.2f}" for n, vs in res.items() for k, v in zip(K_LIST, vs)]
rows += [f"C,{modes[m]},fnd={o['fnd']};half={o['half']};delivered={o['delivered']}" for m, o in out.items()]
open("extended_results.csv", "w").write("\n".join(rows))
print("Saved: 4_optimal_k.png, 5_fault_tolerance.png, 6_multihop_lifetime.png, extended_results.csv")
