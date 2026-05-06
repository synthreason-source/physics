import time
from collections import deque
from pynput import mouse, keyboard

# --- Tunables ---
WINDOW_SECONDS = 10.0      # sliding window length
UPDATE_INTERVAL = 0.2      # how often to recompute & print
LOW_MODULUS_THRESHOLD = 2  # "too efficient" threshold
MIN_KEYS_FOR_TEXT = 3      # ignore windows with almost no keys

# For mouse movement "complexity": bucket into a grid
GRID_SIZE = 200            # pixels per cell (tune for your screen)

# Each event: (timestamp, type, extra)
# type: "click", "key", "move"
# extra for move: (grid_x, grid_y)
events = deque()

last_print = 0.0

# Track last grid position to avoid flooding with tiny moves
last_grid_pos = None

def add_event(ev_type, extra=None):
    now = time.time()
    events.append((now, ev_type, extra))
    prune_old(now)

def prune_old(now=None):
    if now is None:
        now = time.time()
    cutoff = now - WINDOW_SECONDS
    while events and events[0][0] < cutoff:
        events.popleft()

def compute_stats():
    now = time.time()
    prune_old(now)

    clicks = 0
    keys   = 0
    moves  = 0

    # Count unique grid cells touched in the window for movement coverage
    cells = set()

    for t, etype, extra in events:
        if etype == "click":
            clicks += 1
        elif etype == "key":
            keys += 1
        elif etype == "move":
            moves += 1
            if extra is not None:
                cells.add(extra)

    if keys < MIN_KEYS_FOR_TEXT:
        modulus = 0
    else:
        modulus = clicks % keys

    coverage = len(cells)  # how many distinct cells cursor visited
    danger = compute_danger(clicks, keys, moves, coverage, modulus)
    return clicks, keys, moves, coverage, modulus, danger

def compute_danger(clicks, keys, moves, coverage, modulus):
    """
    0–100 danger score.
    - Low modulus (clicks % keys small) is suspicious.
    - Few clicks relative to keys is suspicious.
    - Very little mouse movement (low coverage) while typing is suspicious.
    """
    if keys < MIN_KEYS_FOR_TEXT:
        return 0

    score = 0.0

    # Modulus part
    if modulus <= LOW_MODULUS_THRESHOLD:
        score += 40.0
    elif modulus <= LOW_MODULUS_THRESHOLD + 3:
        score += 20.0

    # Clicks/keys ratio
    ratio_ck = clicks / max(1, keys)
    if ratio_ck < 0.2:
        score += 25.0
    elif ratio_ck < 0.5:
        score += 10.0

    # Mouse movement coverage: lower coverage => more suspicious
    # Normal desktop wandering tends to touch many cells.
    if coverage <= 2:
        score += 25.0
    elif coverage <= 5:
        score += 10.0

    return int(max(0, min(100, score)))

def print_status(clicks, keys, moves, coverage, modulus, danger):
    global last_print
    now = time.time()
    if now - last_print < UPDATE_INTERVAL:
        return
    last_print = now

    bar_len = 20
    filled = int(bar_len * danger / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    status = "LOW "
    if danger > 70:
        status = "HIGH"
    elif danger > 40:
        status = "MED "

    line = (
        f"\rCk={clicks:3d} Ky={keys:3d} Mv={moves:4d} "
        f"Cells={coverage:2d} Mod={modulus:2d} "
        f"DANGER[{bar}] {danger:3d}/100 ({status})"
    )
    print(line, end="", flush=True)

# --- Callbacks ---

def on_mouse_move(x, y):
    global last_grid_pos
    gx = int(x // GRID_SIZE)
    gy = int(y // GRID_SIZE)
    cell = (gx, gy)
    # Only log if we changed cell, to keep data light
    if cell != last_grid_pos:
        last_grid_pos = cell
        add_event("move", cell)
        clicks, keys, moves, coverage, modulus, danger = compute_stats()
        print_status(clicks, keys, moves, coverage, modulus, danger)

def on_mouse_click(x, y, button, pressed):
    if pressed:
        add_event("click")
        clicks, keys, moves, coverage, modulus, danger = compute_stats()
        print_status(clicks, keys, moves, coverage, modulus, danger)

def on_key_press(key):
    add_event("key")
    clicks, keys, moves, coverage, modulus, danger = compute_stats()
    print_status(clicks, keys, moves, coverage, modulus, danger)

def main():
    print("Real-time danger meter (with mouse movement) running (Ctrl+C to exit).")
    print(f"Window = {WINDOW_SECONDS} s, grid cell ~{GRID_SIZE} px.")
    print()

    with mouse.Listener(on_move=on_mouse_move,
                        on_click=on_mouse_click) as m_listener, \
         keyboard.Listener(on_press=on_key_press) as k_listener:
        try:
            while True:
                time.sleep(UPDATE_INTERVAL)
                clicks, keys, moves, coverage, modulus, danger = compute_stats()
                print_status(clicks, keys, moves, coverage, modulus, danger)
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == "__main__":
    main()
