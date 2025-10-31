# -*- coding: utf-8 -*-
import time
import random
import math
import statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple

# ===============================
# Variables globales e instrumentación
# ===============================
prefix_sum_ext: List[int] = []
st_ext: List[int] = []
N_GLOBAL: int = 0

MEMO_WRITES_ARRAY = 0
MEMO_WRITES_HASH = 0

def reset_globals():
    global prefix_sum_ext, st_ext, N_GLOBAL, MEMO_WRITES_ARRAY, MEMO_WRITES_HASH
    prefix_sum_ext = []
    st_ext = []
    N_GLOBAL = 0
    MEMO_WRITES_ARRAY = 0
    MEMO_WRITES_HASH = 0

# ===============================
# Funciones base
# ===============================
def calculate_prefix_sums(st: List[int]) -> None:
    """Precalcula las sumas de prefijos para el arreglo extendido."""
    global prefix_sum_ext, st_ext, N_GLOBAL
    n = len(st) // 2
    N_GLOBAL = 2 * n
    st_ext = st + st
    prefix_sum_ext = [0] * (len(st_ext) + 1)
    for i in range(len(st_ext)):
        prefix_sum_ext[i + 1] = prefix_sum_ext[i] + st_ext[i]

def get_sum(i: int, k: int) -> int:
    """Suma de rango en O(1) usando prefijos."""
    if k == 0:
        return 0
    end = i + k
    return prefix_sum_ext[end] - prefix_sum_ext[i]

# ===============================
# Solución con ARREGLO
# ===============================
def solve_with_array_recursive(i: int, k: int, memo: List[List[int]]) -> int:
    global MEMO_WRITES_ARRAY
    if k <= 0:
        return 0
    if memo[i][k] != -1:
        return memo[i][k]
    n = N_GLOBAL // 2
    if k <= n:
        result = get_sum(i, k)
        memo[i][k] = result
        MEMO_WRITES_ARRAY += 1
        return result
    gain_left = get_sum(i, n)
    remaining_sum_left = get_sum(i + n, k - n)
    opponent_gain_left = solve_with_array_recursive(i + n, k - n, memo)
    val_left = gain_left + remaining_sum_left - opponent_gain_left
    gain_right = get_sum(i + k - n, n)
    remaining_sum_right = get_sum(i, k - n)
    opponent_gain_right = solve_with_array_recursive(i, k - n, memo)
    val_right = gain_right + remaining_sum_right - opponent_gain_right
    result = max(val_left, val_right)
    memo[i][k] = result
    MEMO_WRITES_ARRAY += 1
    return result

def solve_with_array(st: List[int]) -> Tuple[int, int]:
    """Devuelve (máxima satisfacción, número de memoizaciones)."""
    global MEMO_WRITES_ARRAY
    MEMO_WRITES_ARRAY = 0
    n = len(st) // 2
    if n == 0:
        return 0, MEMO_WRITES_ARRAY
    calculate_prefix_sums(st)
    memo = [[-1 for _ in range(2 * n + 1)] for _ in range(4 * n)]
    max_satisfaction = -float('inf')
    for i in range(2 * n):
        gain_first_move = get_sum(i, n)
        remaining_sum = get_sum(i + n, n)
        opponent_gain = solve_with_array_recursive(i + n, n, memo)
        professor_total = gain_first_move + (remaining_sum - opponent_gain)
        max_satisfaction = max(max_satisfaction, professor_total)
    return int(max_satisfaction), MEMO_WRITES_ARRAY

# ===============================
# Solución con HASH TABLE
# ===============================
def solve_with_hashtable_recursive(i: int, k: int, memo: dict) -> int:
    global MEMO_WRITES_HASH
    if k <= 0:
        return 0
    if (i, k) in memo:
        return memo[(i, k)]
    n = N_GLOBAL // 2
    if k <= n:
        result = get_sum(i, k)
        memo[(i, k)] = result
        MEMO_WRITES_HASH += 1
        return result
    gain_left = get_sum(i, n)
    remaining_sum_left = get_sum(i + n, k - n)
    opponent_gain_left = solve_with_hashtable_recursive(i + n, k - n, memo)
    val_left = gain_left + remaining_sum_left - opponent_gain_left
    gain_right = get_sum(i + k - n, n)
    remaining_sum_right = get_sum(i, k - n)
    opponent_gain_right = solve_with_hashtable_recursive(i, k - n, memo)
    val_right = gain_right + remaining_sum_right - opponent_gain_right
    result = max(val_left, val_right)
    memo[(i, k)] = result
    MEMO_WRITES_HASH += 1
    return result

def solve_with_hashtable(st: List[int]) -> Tuple[int, int]:
    """Devuelve (máxima satisfacción, número de memoizaciones)."""
    global MEMO_WRITES_HASH
    MEMO_WRITES_HASH = 0
    n = len(st) // 2
    if n == 0:
        return 0, MEMO_WRITES_HASH
    calculate_prefix_sums(st)
    memo = {}
    max_satisfaction = -float('inf')
    for i in range(2 * n):
        gain_first_move = get_sum(i, n)
        remaining_sum = get_sum(i + n, n)
        opponent_gain = solve_with_hashtable_recursive(i + n, n, memo)
        professor_total = gain_first_move + (remaining_sum - opponent_gain)
        max_satisfaction = max(max_satisfaction, professor_total)
    return int(max_satisfaction), MEMO_WRITES_HASH

# ===============================
# Experimentos de alta precisión
# ===============================
def timed_call(fn, *args, repeats=7):
    """Devuelve mediana de tiempo (s) usando perf_counter_ns para mayor precisión."""
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        out = fn(*args)
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / 1e9)
    return stats.median(durations), out

def run_experiment(n_values, runs_per_n=5, repeats_timer=7, rng_seed=2025):
    """Devuelve tiempos (mediana) y memoizaciones (promedio entero) por n."""
    rnd = random.Random(rng_seed)
    times_array, times_hash = [], []
    memos_array, memos_hash = [], []
    for n_val in n_values:
        t_arr_list, t_hash_list = [], []
        m_arr_list, m_hash_list = [], []
        for _ in range(runs_per_n):
            st = [rnd.randint(-100, 100) for _ in range(2 * n_val)]
            reset_globals()
            t_arr, (_, m_arr) = timed_call(solve_with_array, st, repeats=repeats_timer)
            reset_globals()
            t_hash, (_, m_hash) = timed_call(solve_with_hashtable, st, repeats=repeats_timer)
            t_arr_list.append(t_arr); t_hash_list.append(t_hash)
            m_arr_list.append(m_arr);  m_hash_list.append(m_hash)
        times_array.append(stats.median(t_arr_list))
        times_hash.append(stats.median(t_hash_list))
        memos_array.append(int(stats.mean(m_arr_list)))
        memos_hash.append(int(stats.mean(m_hash_list)))
    return (np.array(times_array), np.array(times_hash),
            np.array(memos_array), np.array(memos_hash))

# ===============================
# Utilidades
# ===============================
def safe_int_input(prompt: str, default: int) -> int:
    try:
        val = int(input(prompt).strip())
        return val
    except Exception:
        print(f"Entrada inválida. Usando valor por defecto {default}.")
        return default

# ===============================
# Ejecución principal (interactiva)
# ===============================
def main():
    print("=== Comparación memoización: Arreglo (morado) vs Hash (rojo) ===")
    n_user = safe_int_input("Ingresa n (número de pares de porciones, entero > 0): ", 200)
    if n_user <= 0:
        n_user = 200

    # Serie 10..n_user (paso 5 para mayor resolución si n_user >= 50)
    step = 5 if n_user >= 50 else 1
    n_values = list(range(step, n_user + 1, step))

    times_array, times_hash, memos_array, memos_hash = run_experiment(
        n_values, runs_per_n=7, repeats_timer=9, rng_seed=2025
    )

    PURPLE = "#6A0DAD"
    RED    = "#D00000"

    # -------- Gráfico 1: Tiempo promedio vs n --------
    plt.figure(figsize=(10, 6), dpi=140)
    plt.plot(n_values, times_array, marker='o', linewidth=2, markersize=4, color=PURPLE, label='Arreglo')
    plt.plot(n_values, times_hash, marker='o', linewidth=2, markersize=4, color=RED,    label='Hash')
    plt.xlabel('n (Número de pares de porciones)')
    plt.ylabel('Tiempo (s) — mediana')
    plt.title('Tiempo de ejecución vs n (mediana, alta precisión)')
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Gráfico 2: Memoizaciones vs n --------
    plt.figure(figsize=(10, 6), dpi=140)
    plt.plot(n_values, memos_array, marker='o', linewidth=2, markersize=4, color=PURPLE, label='Arreglo')
    plt.plot(n_values, memos_hash, marker='o', linewidth=2, markersize=4, color=RED,    label='Hash')
    plt.xlabel('n (Número de pares de porciones)')
    plt.ylabel('Memoizaciones (promedio)')
    plt.title('Crecimiento de memoizaciones vs n')
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Medición puntual para el n ingresado --------
    rnd = random.Random(2025)
    st_big = [rnd.randint(-100, 100) for _ in range(2 * n_user)]

    reset_globals()
    t_arr_big, (_, mem_arr_big) = timed_call(solve_with_array, st_big, repeats=15)

    reset_globals()
    t_hash_big, (_, mem_hash_big) = timed_call(solve_with_hashtable, st_big, repeats=15)

    # Barras de memo y tiempo para n_user
    fig = plt.figure(figsize=(11, 5), dpi=140)
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(['Arreglo', 'Hash'], [mem_arr_big, mem_hash_big], color=[PURPLE, RED])
    ax1.set_title(f'Memoizaciones totales a n = {n_user}')
    ax1.set_ylabel('Memoizaciones')

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(['Arreglo', 'Hash'], [t_arr_big, t_hash_big], color=[PURPLE, RED])
    ax2.set_title(f'Tiempo (s) a n = {n_user} (mediana de 15)')
    ax2.set_ylabel('Tiempo (s)')
    plt.tight_layout()
    plt.show()

    # -------- Tabla comparativa --------
    mejor = 'Arreglo' if t_arr_big < t_hash_big else 'Hash'
    df = pd.DataFrame({
        'Estrategia': ['Arreglo', 'Hash'],
        'Tiempo medido (s)': [t_arr_big, t_hash_big],
        'Memoizaciones': [mem_arr_big, mem_hash_big],
        'Tiempo asintótico': ['O(n^2)', 'O(n^2)'],
        'Espacio asintótico': ['O(n^2)', 'O(n^2)'],
        'Ventajas (prácticas)': [
            'Indexado O(1), sin hashing; excelente constante',
            'Estructura esparsa; sólo guarda estados visitados'
        ],
        'Desventajas (prácticas)': [
            'Matriz densa 4n×(2n+1) (puede usar más RAM)',
            'Overhead de hashing y creación de claves (i,k)'
        ]
    })
    print("\n=== Tabla comparativa (n = {}) ===".format(n_user))
    print(df.to_string(index=False))
    print(f"\n► En esta medición, **{mejor}** fue más rápido para n = {n_user}.")
    print("   Asintóticamente ambas son O(n^2) en tiempo y espacio.\n")

if __name__ == "__main__":
    main()
