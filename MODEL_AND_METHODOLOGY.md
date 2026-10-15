# Model, equations, and interpretation

## 1. Deployment and placement

- Square field: 200 m × 200 m.
- Default sensor nodes: 200.
- 70% of nodes are drawn around three random cluster centers; 30% are uniform background nodes.
- K-Means uses `n_init=10` and a fixed seed for repeatable clustering on the generated deployment.
- Random gateway placement is evaluated across 30 gateway-placement trials per gateway count.

## 2. Radio-energy model

For a packet of `L` bits sent a distance `d`, the first-order radio model is:

\[
E_{TX}(L,d)=L(E_{elec}+\epsilon_{amp}d^2),\qquad E_{RX}(L)=LE_{elec}.
\]

Default values: `L = 4000 bits`, `E_elec = 50 nJ/bit`, `epsilon_amp = 100 pJ/(bit·m²)`, initial sensor battery `E0 = 0.5 J`, and transmission probability `p_tx = 0.5` per node per round.

In `gateway_placement.py`, expected energy per round is the sum of per-packet transmission energy multiplied by `p_tx`. The CSV also records the all-nodes-transmit full-load reference. The lifetime model draws a transmission event only for nodes that are alive; dead nodes do not transmit.

The base placement experiment is a distance/energy baseline and does **not** enforce the 80 m direct-communication cutoff. The extended fault-tolerance and routing experiment does enforce `R_COMM = 80 m` and uses `R_HOP = 40 m` for relay links. Keep this distinction explicit when interpreting or comparing results.

## 3. Metrics

- Average distance: average distance from a sensor to its assigned/nearest gateway.
- Expected energy per round: expected sum of sensor-to-gateway transmit energy at the configured transmission probability (base experiment).
- FND (first node dead): first simulated round where at least one sensor battery reaches zero.
- 50% dead: first round where at least half of the sensors are dead.
- Silhouette score: cluster separation indicator; undefined for `k=1`.
- Cost score: illustrative model `gateway_count × gateway_cost + battery-equivalent cost`. Cost units are arbitrary assumptions, not currency or a procurement quote.
- Delivery after failure: fraction of sensor nodes still having a valid in-range working gateway under the stated primary/backup rule.

## 4. Multi-hop routing

The extended model compares direct transmission, minimum-energy multi-hop routing, and residual-energy-aware multi-hop routing. The routing cost penalizes links/relays according to the configured residual-battery weight. A round is synchronous: packets for a round are processed before depleted nodes are marked unavailable at the round boundary. This is a simplifying assumption.

## 5. Important limitations

1. K-Means minimizes geometric within-cluster squared distance, not the complete network energy objective; it is a baseline, not a globally optimal placement proof.
2. The base and extended experiments have different communication-range assumptions, so their metrics are not directly interchangeable.
3. The cost model is illustrative and omits installation, maintenance, backhaul, gateway power, battery replacement logistics, and discounting.
4. Randomness remains in deployment, placement, and traffic. A single seed supports reproducibility but does not establish statistical significance.
5. The browser dashboard uses a separate JavaScript implementation and is exploratory; its results may differ from the Python simulation.
6. There is no real hardware, field deployment, packet capture, or measured radio-energy data in this project.

## 6. Recommended research-grade extension

For a paper, evaluate at least 20–30 independent deployment seeds; compare K-Means against random, grid, and a clearly defined energy-aware optimizer; enforce one consistent communication model; report mean ± standard deviation or confidence intervals; and include ablation tests for backup gateways and residual-energy-aware routing. State any proposed novelty precisely and compare against published baselines.
