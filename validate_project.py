"""Run reproducibility/smoke checks for the IoT gateway placement project."""
from pathlib import Path
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "gateway_placement.py", "gateway_extended.py", "gateway_sim.html",
    "results.csv", "extended_results.csv",
    "1_placement_comparison.png", "2_energy_vs_k.png", "3_lifetime_vs_k.png",
    "4_optimal_k.png", "5_fault_tolerance.png", "6_multihop_lifetime.png",
]


def run(args):
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    run([sys.executable, "-m", "py_compile", "gateway_placement.py", "gateway_extended.py"])
    run([sys.executable, "gateway_placement.py"])
    run([sys.executable, "gateway_extended.py"])
    missing = [name for name in REQUIRED if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    if missing:
        raise SystemExit("Missing/empty outputs: " + ", ".join(missing))
    for name in ("results.csv", "extended_results.csv"):
        with (ROOT / name).open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            raise SystemExit(f"CSV has no data rows: {name}")
    # PNG signature check (detects truncated/non-PNG output files).
    for name in [x for x in REQUIRED if x.endswith(".png")]:
        if (ROOT / name).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"Invalid PNG output: {name}")
    html = (ROOT / "gateway_sim.html").read_text(encoding="utf-8")
    if "function stats(p,el)" not in html or "Expected energy / round" not in html:
        raise SystemExit("Dashboard checks failed")
    print("\nVALIDATION PASSED: scripts ran and all expected output files were checked.")

if __name__ == "__main__":
    main()
