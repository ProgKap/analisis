import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# For Windows/PowerShell where plots sometimes don't appear:
try:
    matplotlib.use("TkAgg")  # falls back silently if not available
except Exception:
    pass

# =========================
#  Cake Game (2n slices, 1 pedazo por turno)
#  - Profesor (0) empieza, Hermana (1)
#  - αi válido si i está libre y en cada semicírculo (i+1..i+n-1) y (i+n+1..i+2n-1) hay ≥1 libre
#  - Si NO hay jugadas válidas y quedan porciones: el jugador toma TODAS las restantes
#  - Objetivo: maximizar la CANTIDAD de porciones comidas (no satisfacción)
#  - Muestra TODAS las jugadas en UNA sola figura (subplots) + totales
#  - Pide por consola un número PAR y positivo; vuelve a pedir hasta que sea válido
# =========================

def popcount(x: int) -> int:
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


class OneSliceCakeSolver:
    def __init__(self, num_slices: int, use_dict_memo: bool = True):
        if num_slices % 2 != 0 or num_slices <= 0:
            raise ValueError("El número de rebanadas debe ser par y positivo.")
        self.N = num_slices
        self.n = num_slices // 2
        self.use_dict_memo = use_dict_memo
        self.memo = {} if use_dict_memo else [None] * (1 << self.N)

    def _valid_moves(self, mask: int):
        N, n = self.N, self.n
        valid = []
        for i in range(N):
            if mask & (1 << i):
                continue
            side1 = any((mask & (1 << ((i + k) % N))) == 0 for k in range(1, n))
            if not side1:
                continue
            side2 = any((mask & (1 << ((i + n + k) % N))) == 0 for k in range(1, n))
            if side2:
                valid.append(i)
        return valid

    def solve(self, mask: int):
        if self.use_dict_memo:
            hit = self.memo.get(mask)
            if hit is not None:
                return hit
        else:
            if self.memo[mask] is not None:
                return self.memo[mask]

        if mask == (1 << self.N) - 1:
            return (0, -1)

        remaining = self.N - popcount(mask)
        valid = self._valid_moves(mask)

        if not valid:
            res = (remaining, -1)
            if self.use_dict_memo: self.memo[mask] = res
            else: self.memo[mask] = res
            return res

        best_cnt, best_move = -1, -1
        for mv in valid:
            next_mask = mask | (1 << mv)
            opp_cnt, _ = self.solve(next_mask)
            my_total = 1 + ((remaining - 1) - opp_cnt)
            if my_total > best_cnt:
                best_cnt, best_move = my_total, mv

        res = (best_cnt, best_move)
        if self.use_dict_memo: self.memo[mask] = res
        else: self.memo[mask] = res
        return res


def run_game_simulation(num_slices: int):
    solver = OneSliceCakeSolver(num_slices=num_slices, use_dict_memo=True)
    N = solver.N
    mask = 0
    turn = 0
    counts = [0, 0]                  # (prof, hermana)
    eaten_by = [-1] * N              # -1 libre, 0 prof, 1 hermana
    history = []

    while mask != (1 << N) - 1:
        turn += 1
        player = (turn - 1) % 2

        valid = solver._valid_moves(mask)
        if not valid:
            eaten_now = []
            for i in range(N):
                if (mask & (1 << i)) == 0:
                    mask |= (1 << i)
                    eaten_by[i] = player
                    eaten_now.append(i)
            counts[player] += len(eaten_now)
            history.append({
                'turn': turn, 'player': player, 'move': -1,
                'eaten_now': eaten_now, 'counts': list(counts), 'eaten_by': list(eaten_by)
            })
            break
        else:
            _, mv = solver.solve(mask)
            mask |= (1 << mv)
            eaten_by[mv] = player
            counts[player] += 1
            history.append({
                'turn': turn, 'player': player, 'move': mv,
                'eaten_now': [mv], 'counts': list(counts), 'eaten_by': list(eaten_by)
            })

    return history


def draw_all_turns_one_figure(history, num_slices: int):
    N = num_slices
    slice_angle = 2 * np.pi / N
    T = len(history)
    cols = min(6, T)
    rows = (T + cols - 1) // cols

    final_prof, final_sis = history[-1]['counts']

    fig = plt.figure(figsize=(4.8 * cols, 5.1 * rows))
    fig.suptitle("Juego de la Torta – Todas las jugadas (1 pedazo por turno)", fontsize=16, y=0.99)

    fig.text(
        0.5, 0.965,
        f"Totales → Profesor Mladen: {final_prof}   |   Hermana: {final_sis}   |   Total: {final_prof + final_sis}/{N}",
        ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f1f1f1", ec="#888")
    )
    fig.text(
        0.5, 0.943,
        "Colores: Profesor (rojo) · Hermana (azul) · Libre (gris)  —  Borde verde: pedazo tomado en esa jugada",
        ha='center', va='center', fontsize=10, color="#444"
    )

    for idx, step in enumerate(history, 1):
        ax = fig.add_subplot(rows, cols, idx, projection='polar')
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_yticklabels([])

        centers = np.arange(N) * slice_angle + slice_angle / 2
        ax.set_xticks(centers)
        ax.set_xticklabels([f"α{i}" for i in range(N)], fontsize=8 if N > 16 else 9)

        widths = np.full(N, slice_angle)
        heights = np.ones(N)
        colors = []
        for i in range(N):
            who = step['eaten_by'][i]
            if who == 0: colors.append('#f94144')   # Profesor
            elif who == 1: colors.append('#577590') # Hermana
            else: colors.append('#d9d9d9')          # Libre

        bars = ax.bar(
            x=np.arange(N) * slice_angle, height=heights, width=widths, bottom=0,
            align='edge', edgecolor='black', linewidth=1.0, color=colors, alpha=0.9
        )

        for j in step['eaten_now']:
            bars[j].set_linewidth(3.0)
            bars[j].set_edgecolor('green')

        who_text = "Profesor" if step['player'] == 0 else "Hermana"
        ax.set_title(f"Jugada {step['turn']} – {who_text}", fontsize=11, pad=8)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.925])
    plt.show(block=True)  # BLOQUEA hasta cerrar la ventana (evita que "no aparezca" en consola)


# =========================
#           MAIN
# =========================
if __name__ == "__main__":
    # Reintenta hasta recibir un PAR válido (así no te quedas en el prompt del sistema)
    while True:
        try:
            n_input = input("Ingrese el número de pedazos de la torta (PAR y > 0): ").strip()
            N = int(n_input)
            if N > 0 and N % 2 == 0:
                break
            print("El número debe ser PAR y positivo. Intente nuevamente.")
        except Exception:
            print("Entrada inválida. Intente nuevamente.")

    history = run_game_simulation(num_slices=N)
    draw_all_turns_one_figure(history, num_slices=N)

    final_prof, final_sis = history[-1]['counts']
    print(f"\nResultado final → Profesor Mladen: {final_prof} | Hermana: {final_sis} (total {final_prof+final_sis}/{N})")
