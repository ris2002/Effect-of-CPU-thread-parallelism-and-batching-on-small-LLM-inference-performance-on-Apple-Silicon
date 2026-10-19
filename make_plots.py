
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Colours: categorical slots in fixed order, plus recessive ink for ideal/grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e4e3dc"


def load_runs(path):
    runs = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            run = int(row["run"])
            stdev = row.get("stdev_s") or ""
            runs.setdefault(run, []).append({
                "threads": int(row["threads"]),
                "mean": float(row["mean_s"]),
                "stdev": float(stdev) if stdev.strip() else None,
            })
    for rows in runs.values():
        rows.sort(key=lambda r: r["threads"])
    return runs


def add_metrics(rows):
    """Speedup, efficiency and Amdahl serial fraction, from unrounded means."""
    t1 = next(r["mean"] for r in rows if r["threads"] == 1)
    for r in rows:
        n = r["threads"]
        s = t1 / r["mean"]
        r["speedup"] = s
        r["efficiency"] = s / n * 100
        r["f"] = ((n / s - 1) / (n - 1) * 100) if n > 1 else None
    return rows


def style(ax, xlabel, ylabel, threads):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xticks(threads)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def plot_speedup(rows, out, title):
    th = [r["threads"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(th, th, linestyle="--", linewidth=1.5, color=MUTED, label="Ideal (S = N)")
    ax.plot(th, [r["speedup"] for r in rows], marker="o", markersize=6,
            linewidth=2, color=SERIES[0], label="Measured")
    style(ax, "Threads (N)", "Speedup (S = T1 / TN)", th)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(frameon=False)
    save(fig, out)


def plot_single(rows, key, ylabel, out, title, ylim=None):
    pts = [r for r in rows if r[key] is not None]
    th = [r["threads"] for r in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(th, [r[key] for r in pts], marker="o", markersize=6,
            linewidth=2, color=SERIES[0])
    style(ax, "Threads (N)", ylabel, [r["threads"] for r in rows])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, color=INK, fontsize=11)
    save(fig, out)


def plot_compare(runs, run_ids, out):
    if len(run_ids) > len(SERIES):
        raise SystemExit(f"Compare at most {len(SERIES)} runs per chart.")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_threads = sorted({r["threads"] for rid in run_ids for r in runs[rid]})
    ax.plot(all_threads, all_threads, linestyle="--", linewidth=1.5,
            color=MUTED, label="Ideal (S = N)")
    for colour, rid in zip(SERIES, run_ids):
        rows = runs[rid]
        ax.plot([r["threads"] for r in rows], [r["speedup"] for r in rows],
                marker="o", markersize=6, linewidth=2, color=colour,
                label=f"Run {rid}")
    style(ax, "Threads (N)", "Speedup (S = T1 / TN)", all_threads)
    ax.set_title("Speedup across runs", color=INK, fontsize=11)
    ax.legend(frameon=False)
    save(fig, out)


def write_summary(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "mean_s", "stdev_s", "speedup",
                    "efficiency_pct", "serial_fraction_pct"])
        for r in rows:
            w.writerow([
                r["threads"],
                f"{r['mean']:.4f}",
                "" if r["stdev"] is None else f"{r['stdev']:.4f}",
                f"{r['speedup']:.2f}",
                f"{r['efficiency']:.1f}",
                "" if r["f"] is None else f"{r['f']:.2f}",
            ])
    print("saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="results CSV (run, date, conditions, threads, mean_s, stdev_s)")
    p.add_argument("--run", type=int, required=True, help="run number to plot")
    p.add_argument("--compare", type=int, nargs="*", help="run numbers to overlay (max 4)")
    args = p.parse_args()

    runs = {rid: add_metrics(rows) for rid, rows in load_runs(args.csv).items()}
    if args.run not in runs:
        raise SystemExit(f"Run {args.run} not found. Runs in file: {sorted(runs)}")

    out_dir = Path(args.csv).resolve().parent
    rows = runs[args.run]
    tag = f"run{args.run}"

    write_summary(rows, out_dir / f"{tag}_summary.csv")
    plot_speedup(rows, out_dir / f"{tag}_speedup.png",
                 f"Thread scaling of exp (500M float32), run {args.run}")
    plot_single(rows, "efficiency", "Parallel efficiency (%)",
                out_dir / f"{tag}_efficiency.png",
                f"Parallel efficiency, run {args.run}", ylim=(0, 110))
    plot_single(rows, "f", "Implied serial fraction f (%)",
                out_dir / f"{tag}_serial_fraction.png",
                f"Amdahl serial fraction, run {args.run}", ylim=(0, None))

    if args.compare:
        missing = [r for r in args.compare if r not in runs]
        if missing:
            raise SystemExit(f"Runs not found: {missing}")
        plot_compare(runs, args.compare, out_dir / "runs_compare_speedup.png")


if __name__ == "__main__":
    main()
