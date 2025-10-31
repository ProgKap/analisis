import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEED = 2025

def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

@dataclass
class PrefixRange:
    base: List[int]
    ext: List[int]
    pref: List[int]
    m: int
    ext_m: int

    @staticmethod
    def build(st: List[int]) -> "PrefixRange":
        m = len(st)
        ext = st + st
        pref = [0] * (len(ext) + 1)
        for i, v in enumerate(ext):
            pref[i + 1] = pref[i] + v
        return PrefixRange(base=st, ext=ext, pref=pref, m=m, ext_m=len(ext))

    def range_sum(self, i: int, k: int) -> int:
        if k <= 0:
            return 0
        end = i + k
        return self.pref[end] - self.pref[i]

def _solve_rec_array(i: int, k: int, n: int, pr: PrefixRange, memo: List[List[int]]) -> int:
    if k <= 0:
        return 0
    if memo[i][k] != -1:
        return memo[i][k]
    if k <= n:
        res = pr.range_sum(i, k)
        memo[i][k] = res
        return res
    gain_left = pr.range_sum(i, n)
    rem_left = pr.range_sum(i + n, k - n)
    opp_left = _solve_rec_array(i + n, k - n, n, pr, memo)
    val_left = gain_left + rem_left - opp_left
    gain_right = pr.range_sum(i + k - n, n)
    rem_right = pr.range_sum(i, k - n)
    opp_right = _solve_rec_array(i, k - n, n, pr, memo)
    val_right = gain_right + rem_right - opp_right
    res = max(val_left, val_right)
    memo[i][k] = res
    return res

def solve_with_array(st: List[int]) -> int:
    m = len(st)
    n = m // 2
    if n == 0:
        return 0
    pr = PrefixRange.build(st)
    memo = [[-1 for _ in range(2 * n + 1)] for _ in range(4 * n)]
    best = -10**18
    for i in range(2 * n):
        gain_first = pr.range_sum(i, n)
        rem = pr.range_sum(i + n, n)
        opp = _solve_rec_array(i + n, n, n, pr, memo)
        prof = gain_first + (rem - opp)
        best = max(best, prof)
    return best

def _solve_rec_hash(i: int, k: int, n: int, pr: PrefixRange, memo: Dict[Tuple[int, int], int]) -> int:
    if k <= 0:
        return 0
    key = (i, k)
    if key in memo:
        return memo[key]
    if k <= n:
        res = pr.range_sum(i, k)
        memo[key] = res
        return res
    gain_left = pr.range_sum(i, n)
    rem_left = pr.range_sum(i + n, k - n)
    opp_left = _solve_rec_hash(i + n, k - n, n, pr, memo)
    val_left = gain_left + rem_left - opp_left
    gain_right = pr.range_sum(i + k - n, n)
    rem_right = pr.range_sum(i, k - n)
    opp_right = _solve_rec_hash(i, k - n, n, pr, memo)
    val_right = gain_right + rem_right - opp_right
    res = max(val_left, val_right)
    memo[key] = res
    return res

def solve_with_hashtable(st: List[int]) -> int:
    m = len(st)
    n = m // 2
    if n == 0:
        return 0
    pr = PrefixRange.build(st)
    memo: Dict[Tuple[int, int], int] = {}
    best = -10**18
    for i in range(2 * n):
        gain_first = pr.range_sum(i, n)
        rem = pr.range_sum(i + n, n)
        opp = _solve_rec_hash(i + n, n, n, pr, memo)
        prof = gain_first + (rem - opp)
        best = max(best, prof)
    return best

def _time_run(st: List[int]) -> Tuple[float, float]:
    t0 = time.perf_counter()
    solve_with_array(st)
    t1 = time.perf_counter()
    solve_with_hashtable(st)
    t2 = time.perf_counter()
    return (t1 - t0, t2 - t1)

def _gen_instance(n: int, lo: int, hi: int) -> List[int]:
    return [random.randint(lo, hi) for _ in range(2 * n)]

def _aggregate(xs, median=False):
    xs = list(xs)
    return statistics.median(xs) if median else (sum(xs) / len(xs))

def run_experiment_1(n_values, reps, lo, hi, median, out_csv, out_fig):
    print("\\n=== Experimento 1: tiempo vs n ===\\n")
    set_seed()
    times_a, times_h = [], []
    rows = []
    for n in n_values:
        ra, rh = [], []
        for _ in range(reps):
            st = _gen_instance(n, lo, hi)
            ta, th = _time_run(st)
            ra.append(ta); rh.append(th)
        A = _aggregate(ra, median); H = _aggregate(rh, median)
        times_a.append(A); times_h.append(H)
        rows.append([n, A, H])
        print(f"n={n:3d} | arreglo={A:.6f} s | hash={H:.6f} s")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n","time_array_s","time_hash_s","reps","lo","hi","stat"])
            stat = "median" if median else "mean"
            for n,(A,H) in zip(n_values, zip(times_a,times_h)):
                w.writerow([n, f"{A:.8f}", f"{H:.8f}", reps, lo, hi, stat])
    plt.figure(figsize=(10,6))
    plt.plot(n_values, times_a, "o-", label="Memo arreglo")
    plt.plot(n_values, times_h, "x-", label="Memo hash")
    plt.xlabel("n (número de pares de porciones)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de estrategias de memoización")
    plt.grid(True); plt.legend(); plt.tight_layout()
    if out_fig:
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=150)
    plt.show()
    return times_a, times_h

def run_experiment_2(n_values, reps, lo, hi, median, exp_arr, exp_hash, out_fig):
    print("\\n=== Experimento 2: empírico vs guía ===\\n")
    set_seed()
    times_a, times_h = [], []
    for n in n_values:
        ra, rh = [], []
        for _ in range(reps):
            st = _gen_instance(n, lo, hi)
            ta, th = _time_run(st)
            ra.append(ta); rh.append(th)
        A = _aggregate(ra, median); H = _aggregate(rh, median)
        times_a.append(A); times_h.append(H)
        print(f"n={n:3d} | arreglo={A:.6f} s | hash={H:.6f} s")
    max_n = max(n_values) if n_values else 1
    guide_a = [max(times_a) * (n/max_n)**exp_arr for n in n_values]
    guide_h = [max(times_h) * (n/max_n)**exp_hash for n in n_values]
    plt.figure(figsize=(10,6))
    plt.plot(n_values, times_a, "o-", label="Empírico (arreglo)")
    plt.plot(n_values, guide_a, "--", label=f"Guía ~O(n^{exp_arr})")
    plt.plot(n_values, times_h, "x-", label="Empírico (hash)")
    plt.plot(n_values, guide_h, "--", label=f"Guía ~O(n^{exp_hash})")
    plt.xlabel("n (número de pares de porciones)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Crecimiento empírico vs curvas guía")
    plt.grid(True); plt.legend(); plt.tight_layout()
    if out_fig:
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=150)
    plt.show()

def self_test():
    st = [3, -1, 5, 2]
    a = solve_with_array(st)
    h = solve_with_hashtable(st)
    assert a == h, f"Discrepancia: array={a}, hash={h}"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-start", type=int, default=10)
    p.add_argument("--n-end", type=int, default=400)
    p.add_argument("--n-step", type=int, default=10)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--lo", type=int, default=-100)
    p.add_argument("--hi", type=int, default=100)
    p.add_argument("--median", action="store_true")
    p.add_argument("--csv", default="out/exp1_resultados.csv")
    p.add_argument("--fig1", default="out/exp1_tiempo_vs_n.png")
    p.add_argument("--fig2", default="out/exp2_empirico_vs_guia.png")
    p.add_argument("--exp-arr", type=int, default=2)
    p.add_argument("--exp-hash", type=int, default=2)
    args = p.parse_args()
    set_seed(args.seed)
    self_test()
    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))
    run_experiment_1(n_values, args.reps, args.lo, args.hi, args.median, args.csv, args.fig1)
    run_experiment_2(n_values, args.reps, args.lo, args.hi, args.median, args.exp_arr, args.exp_hash, args.fig2)