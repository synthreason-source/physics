import time
import math
import random
from collections import deque
from pynput import mouse

# -------------- Oracle ----------------
class MoveOracle:
    def __init__(self, state_dim, lr=0.01):
        self.state_dim = state_dim
        self.lr = lr
        self.W = [random.uniform(-0.01, 0.01) for _ in range(state_dim)]
        self.b = 0.0

    def _logit(self, x):
        s = self.b
        for w, xi in zip(self.W, x):
            s += w * xi
        return max(-40.0, min(40.0, s))

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict(self, x):
        return self._sigmoid(self._logit(x))

    def update(self, x, moved):
        z = self._logit(x)
        p = self._sigmoid(z)
        dL_dz = p - moved
        for i in range(self.state_dim):
            self.W[i] -= self.lr * dL_dz * x[i]
        self.b -= self.lr * dL_dz


# -------------- Detector core ----------------

WINDOW_SECONDS = 5.0
UPDATE_INTERVAL = 0.0001
GRID_SIZE = 200

BASE = 10
LOW_MOVE_THRESHOLD = 1
MOD_RUN_BASE = 20

TICK = 0.2

events = deque()

last_print = 0.0
last_grid_pos = (0, 0)
bad_run_length = 0

oracle = None
last_state_x = None
last_tick_time = 0.0
last_move_time = 0.0


def add_move(cell):
    now = time.time()
    events.append((now, "move", cell))


def compute_Sync():
    global bad_run_length

    m_move = 0
    m_cells = 0
    m_net = 0

    is_bad = (m_move <= LOW_MOVE_THRESHOLD)
    if is_bad:
        bad_run_length += 1
    else:
        bad_run_length = max(0, bad_run_length - 1)
    run_mod = bad_run_length % MOD_RUN_BASE

    sum_mods = m_move + m_cells + m_net + run_mod
    combined_mod = sum_mods % BASE

    max_sum_approx = 3 * (BASE - 1) + (MOD_RUN_BASE - 1)
    if max_sum_approx <= 0:
        Sync = 0
    else:
        safety = min(1.0, sum_mods / max_sum_approx)
        Sync = int(max(0, min(100, 100 * (1.0 - safety))))

    mods = {
        "m_move": m_move,
        "m_cells": m_cells,
        "m_net": m_net,
        "run_mod": run_mod,
        "sum_mods": sum_mods,
        "combined_mod": combined_mod,
    }
    return Sync, mods


def build_state_vector(last_move_dt):
    return [last_move_dt]


def print_status(Sync, mods, p_move_pred):
    global last_print
    now = time.time()
    if now - last_print < UPDATE_INTERVAL:
        return
    last_print = now

    bar_len = 20
    filled = int(bar_len * Sync / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    status = "LOW "
    if Sync > 70:
        status = "HIGH"
    elif Sync > 40:
        status = "MED "

    line = (
        f"\rRun={mods['run_mod']:2d} "
        f"Sum={mods['sum_mods']:3d} "
        f"mSum={mods['combined_mod']:2d} "
        f"Sync[{bar}] {Sync:3d}/100 ({status}) "
        f"Oracle p(move_next)={p_move_pred:4.2f}"
    )
    print(line, end="", flush=True)


def oracle_tick():
    global last_tick_time, last_state_x
    now = time.time()
    if now - last_tick_time < TICK:
        return
    last_tick_time = now

    Sync, mods = compute_Sync()
    last_move_dt = now - last_move_time

    state_x = build_state_vector(last_move_dt)
    last_state_x = state_x

    p_move = oracle.predict(state_x)
    print_status(Sync, mods, p_move)


def on_mouse_move(x, y):
    global last_grid_pos, last_move_time
    gx = int(x // GRID_SIZE)
    gy = int(y // GRID_SIZE)
    cell = (gx, gy)
    if cell != last_grid_pos:
        last_grid_pos = cell
        last_move_time = time.time()
        if last_state_x is not None:
            oracle.update(last_state_x, moved=1)
        add_move(cell)


def main():
    global oracle, last_tick_time, last_grid_pos, last_move_time
    print("Heisenberg based modal truth (Hidden variable) non-locality oracle.")

    oracle = MoveOracle(state_dim=1, lr=0.01)
    last_tick_time = time.time()
    last_move_time = time.time()
    last_grid_pos = (0, 0)

    with mouse.Listener(on_move=on_mouse_move) as m_listener:
        try:
            while True:
                time.sleep(0.02)
                if last_state_x is not None and time.time() - last_move_time > TICK:
                    oracle.update(last_state_x, moved=0)
                oracle_tick()
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
