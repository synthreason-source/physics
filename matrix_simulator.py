import time
from collections import deque
from pynput import mouse, keyboard

# --- Tunables ---
WINDOW_SECONDS = 10.0      # sliding window length
UPDATE_INTERVAL = 0.2      # how often to recompute & print
LOW_MODULUS_THRESHOLD = 2  # "too efficient" threshold
MIN_KEYS_FOR_TEXT = 3      # ignore windows with almost no keys

# Each event: (timestamp, type) where type is "click" or "key"
events = deque()

last_print = 0.0

def add_event(ev_type):
    now = time.time()
    events.append((now, ev_type))
    prune_old(now)

def prune_old(now=None):
    if now is None:
        now = time.time()
    # Drop events older than WINDOW_SECONDS
    cutoff = now - WINDOW_SECONDS
    while events and events[0][0] < cutoff:
        events.popleft()

def compute_stats():
    now = time.time()
    prune_old(now)

    clicks = sum(1 for t, e in events if e == "click")
    keys   = sum(1 for t, e in events if e == "key")

    if keys < MIN_KEYS_FOR_TEXT:
        modulus = 0
    else:
        modulus = clicks % keys

    danger = compute_danger(clicks, keys, modulus)
    return clicks, keys, modulus, danger

def compute_danger(clicks, keys, modulus):
    """
    0–100 danger score, updated continuously.
    - Low modulus (<= threshold) => more suspicious.
    - Few clicks relative to keys => more suspicious.
    """
    if keys < MIN_KEYS_FOR_TEXT:
        return 0

    score = 0.0

    # Modulus part
    if modulus <= LOW_MODULUS_THRESHOLD:
        score += 60.0
    elif modulus <= LOW_MODULUS_THRESHOLD + 3:
        score += 30.0

    # Clicks/keys ratio
    ratio = clicks / max(1, keys)
    if ratio < 0.2:
        score += 40.0
    elif ratio < 0.5:
        score += 20.0

    return int(max(0, min(100, score)))

def print_status(clicks, keys, modulus, danger):
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

    # \r + flush to behave like a live status line
    line = (
        f"\rClicks={clicks:3d}  Keys={keys:3d}  "
        f"Mod={modulus:2d}  DANGER[{bar}] {danger:3d}/100 ({status})"
    )
    print(line, end="", flush=True)

# --- Callbacks for listeners ---

def on_mouse_click(x, y, button, pressed):
    if pressed:
        add_event("click")
        clicks, keys, modulus, danger = compute_stats()
        print_status(clicks, keys, modulus, danger)

def on_key_press(key):
    # Count every key press as text-related for now
    add_event("key")
    clicks, keys, modulus, danger = compute_stats()
    print_status(clicks, keys, modulus, danger)

def main():
    print("Real-time danger meter running (Ctrl+C to exit).")
    print("Window =", WINDOW_SECONDS, "seconds")
    print()

    with mouse.Listener(on_click=on_mouse_click) as m_listener, \
         keyboard.Listener(on_press=on_key_press) as k_listener:
        try:
            while True:
                # Even if no events, periodically decay/prune and show idle state
                time.sleep(UPDATE_INTERVAL)
                clicks, keys, modulus, danger = compute_stats()
                print_status(clicks, keys, modulus, danger)
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == "__main__":
    main()
