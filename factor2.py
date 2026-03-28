import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The "curve" in the metamaterial simulation is the hyperbola defined by x * y = N.
# We can calculate all primes up to a certain limit by checking if this continuous curve 
# intersects any discrete integer coordinates (x, y) where x > 1 and y > 1.

def calculate_primes_via_curve(limit):
    primes = []
    # We will also keep track of the intersections for visualization
    intersections = {}
    
    for N in range(2, limit + 1):
        is_prime = True
        curve_points = []
        
        # We only need to check x up to sqrt(N) due to the symmetry of x * y = N
        max_x = int(np.sqrt(N))
        
        for x in range(2, max_x + 1):
            # The curve equation: y = N / x
            y = N / x
            
            # If y is a perfect integer, the curve intersects a metamaterial well
            if y.is_integer():
                is_prime = False
                curve_points.append((x, int(y)))
                # Add the symmetric point as well
                if x != int(y):
                    curve_points.append((int(y), x))
                    
        if is_prime:
            primes.append(N)
        else:
            intersections[N] = curve_points
            
    return primes, intersections

# Let's calculate primes up to 1000 using this curve method
max_N = 1000
computed_primes, composite_intersections = calculate_primes_via_curve(max_N)

# Save the primes to a CSV
df_primes = pd.DataFrame({"Prime": computed_primes})
df_primes.to_csv("output/curve_primes.csv", index=False)

# Let's visualize this concept using Matplotlib
# We will plot the grid and the hyperbola for a composite (e.g., 24) and a prime (e.g., 23)
fig, ax = plt.subplots(figsize=(8, 8))

# Define grid limits for visualization
grid_max = 25
x_grid = np.arange(1, grid_max + 1)
y_grid = np.arange(1, grid_max + 1)
X, Y = np.meshgrid(x_grid, y_grid)

# Plot the internal integer grid (excluding x=1 and y=1 to represent non-trivial factors)
ax.scatter(X[1:, 1:], Y[1:, 1:], color='lightgray', s=10, label='Internal Integer Grid (Wells)')

# Plot continuous curves for x * y = N
x_cont = np.linspace(1, grid_max, 500)

# 1. Composite Number Curve (N=24)
N_comp = 24
y_comp = N_comp / x_cont
# Filter points within grid
valid_comp = y_comp <= grid_max
ax.plot(x_cont[valid_comp], y_comp[valid_comp], color='#ffaa00', linewidth=2, label=f'Composite Curve ($x \\cdot y = {N_comp}$)')

# Highlight intersections for the composite curve
comp_x = [p[0] for p in composite_intersections[N_comp]]
comp_y = [p[1] for p in composite_intersections[N_comp]]
ax.scatter(comp_x, comp_y, color='red', s=80, zorder=5, label='Internal Intersections (Factors)')

# 2. Prime Number Curve (N=23)
N_prime = 23
y_prime = N_prime / x_cont
# Filter points within grid
valid_prime = y_prime <= grid_max
ax.plot(x_cont[valid_prime], y_prime[valid_prime], color='#00ffcc', linewidth=2, label=f'Prime Curve ($x \\cdot y = {N_prime}$)')

ax.set_xlim(1, grid_max)
ax.set_ylim(1, grid_max)
ax.set_aspect('equal')
ax.set_title("Hyperbolic Curve Sieve: Prime vs Composite", fontsize=14)
ax.set_xlabel("Grid Factor X")
ax.set_ylabel("Grid Factor Y")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

plt.tight_layout()
fig.savefig("output/hyperbola_sieve.png", dpi=150)

# Print first 50 primes found
print(f"Calculated {len(computed_primes)} primes up to {max_N} using the curve method.")
print(f"First 50 primes: {computed_primes[:50]}")