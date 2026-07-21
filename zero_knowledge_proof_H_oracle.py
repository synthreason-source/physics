import time
import math
import random
from collections import deque
from pynput import mouse
import psutil

# -------------- Oracle ----------------
# something magic, based photon-matter sync detector

class MoveOracle:
    def __init__(self, state_dim, lr=0.01):
        self.state_dim = state_dim
        self.lr = lr
        # single output logit: "will move in next horizon"
        self.W = [random.uniform(-0.01, 0.01) for _ in range(state_dim)]
        self.b = 0.0

    def _logit(self, x):
        s = self.b
        for w, xi in zip(self.W, x):
            s += w * xi
        # clip for numeric stability
        return max(-40.0, min(40.0, s))

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def predict(self, x):
        z = self._logit(x)
        return self._sigmoid(z)

    def update(self, x, moved):
        z = self._logit(x)
        p = self._sigmoid(z)
        # derivative of BCE wrt logit
        dL_dz = p - moved
        for i in range(self.state_dim):
            self.W[i] -= self.lr * dL_dz * x[i]
        self.b -= self.lr * dL_dz


# -------------- Detector core ----------------

WINDOW_SECONDS = 10.0
UPDATE_INTERVAL = 0.1
NET_SAMPLE_INTERVAL = 0.5
GRID_SIZE = 200

BASE = 10              # modulus base for all terms
LOW_MOVE_THRESHOLD = 1 # "low movement" in modulus space
MOD_RUN_BASE = 20

TICK = 0.2             # oracle horizon

# events: (t, type, cell) type == "move"
events = deque()
net_samples = deque()

last_net_check = 0.0
last_net_counters = None

last_print = 0.0
last_grid_pos = (0, 0)
bad_run_length = 0

# oracle state
oracle = None
last_state_x = None
last_tick_time = 0.0
last_move_time = 0.0


def prune_old(now=None):
    if now is None:
        now = time.time()
    cutoff = now - WINDOW_SECONDS
    while events and events[0][0] < cutoff:
        events.popleft()
    while net_samples and net_samples[0][0] < cutoff:
        net_samples.popleft()


def sample_network():
    global last_net_counters, last_net_check
    now = time.time()
    if now - last_net_check < NET_SAMPLE_INTERVAL:
        return
    last_net_check = now

    counters = psutil.net_io_counters()
    if last_net_counters is not None:
        ps_delta = counters.packets_sent - last_net_counters.packets_sent
        pr_delta = counters.packets_recv - last_net_counters.packets_recv
        net_samples.append((now, ps_delta, pr_delta))
        prune_old(now)
    last_net_counters = counters


def add_move(cell):
    now = time.time()
    events.append((now, "move", cell))
    prune_old(now)


def compute_stats():
    now = time.time()
    prune_old(now)

    moves = 0
    cells = set()

    for t, etype, cell in events:
        if etype == "move":
            moves += 1
            if cell is not None:
                cells.add(cell)

    coverage = len(cells)

    total_ps = sum(s for _, s, _ in net_samples)
    total_pr = sum(r for _, _, r in net_samples)
    window_len = WINDOW_SECONDS if net_samples else 1.0
    pkts_per_sec = (total_ps + total_pr) / max(1.0, window_len)

    Sync, mods = compute_Sync(moves, coverage, pkts_per_sec)
    return moves, coverage, pkts_per_sec, Sync, mods


def compute_Sync(moves, coverage, pkts_per_sec):
    global bad_run_length

    # base moduli, all relative to fixed BASE
    m_move = moves % BASE
    m_cells = coverage % BASE
    m_net = int(pkts_per_sec) % BASE

    # define "low movement" window via m_move
    is_bad = (m_move <= LOW_MOVE_THRESHOLD)
    if is_bad:
        bad_run_length += 1
    else:
        bad_run_length = max(0, bad_run_length - 1)
    run_mod = bad_run_length % MOD_RUN_BASE

    # sum all moduli
    sum_mods = m_move + m_cells + m_net + run_mod
    combined_mod = sum_mods % BASE

    # theoretical max
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


def build_state_vector(moves, coverage, pkts_per_sec, last_move_dt):
    # simple scaling
    return [
        moves / 100.0,
        coverage / 100.0,
        pkts_per_sec / 100.0,
        last_move_dt,
    ]


def print_status(moves, coverage, pkts_per_sec, Sync, mods, p_move_pred):
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
        f"\rMv={moves:4d} Cells={coverage:3d} "
        f"Pkts/s={pkts_per_sec:5.1f} "
        f"mMv={mods['m_move']:2d} mCells={mods['m_cells']:2d} "
        f"mNet={mods['m_net']:2d} Run={mods['run_mod']:2d} "
        f"Sum={mods['sum_mods']:3d} mSum={mods['combined_mod']:2d} "
        f"Sync[{bar}] {Sync:3d}/100 ({status}) "
        f"Oracle p(move_next)={p_move_pred:4.2f}"
    )
    print(line, end="", flush=True)


# ---------- Oracle tick + mouse callback ----------

def oracle_tick():
    global last_tick_time, last_state_x
    now = time.time()
    if now - last_tick_time < TICK:
        return
    last_tick_time = now

    moves, coverage, pkts, Sync, mods = compute_stats()
    last_move_dt = now - last_move_time

    state_x = build_state_vector(moves, coverage, pkts, last_move_dt)
    last_state_x = state_x

    p_move = oracle.predict(state_x)
    print_status(moves, coverage, pkts, Sync, mods, p_move)


def on_mouse_move(x, y):
    global last_grid_pos, last_move_time
    gx = int(x // GRID_SIZE)
    gy = int(y // GRID_SIZE)
    cell = (gx, gy)
    if cell != last_grid_pos:
        last_grid_pos = cell
        last_move_time = time.time()
        # positive oracle example: movement occurred
        if last_state_x is not None:
            oracle.update(last_state_x, moved=1)
        add_move(cell)
        sample_network()


# -------------- main() ----------------

def main():
    global oracle, last_tick_time, last_grid_pos, last_move_time
    print("Mouse+network modulus detector with movement oracle.")
    print("No keyboard or clicks used. Ctrl+C to exit.\n")

    state_dim = 4
    oracle_lr = 0.01
    oracle = MoveOracle(state_dim=state_dim, lr=oracle_lr)

    last_tick_time = time.time()
    last_move_time = time.time()
    last_grid_pos = (0, 0)

    # run mouse listener and periodic oracle updates
    with mouse.Listener(on_move=on_mouse_move) as m_listener:
        try:
            while True:
                time.sleep(0.02)
                sample_network()
                # negative oracle example: if no movement happened this tick
                if last_state_x is not None:
                    # only update negatively when we pass a tick without new move
                    if time.time() - last_move_time > TICK:
                        oracle.update(last_state_x, moved=0)
                oracle_tick()
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()