import numpy as np
import math
import warnings
import sys
import random  # For randomization

sys.setrecursionlimit(2000)
warnings.filterwarnings("ignore")

# ==========================================
# 1. ASTRONOMICAL PHYSICS KERNEL
# ==========================================

class AstroDomain:
    def __init__(self, name, initial_scale=10.0):
        self.name = name
        # Initialize to scale for better convergence on balanced factors
        self.val = initial_scale
        self.velocity = 0.0
        
    def update_multiplicative(self, factor, dt):
        """
        Updates value by a multiplicative factor (safe for huge numbers).
        val_new = val_old * (1 + speed * dt)
        """
        # Damping on the factor itself to prevent oscillation
        # We treat 'velocity' here as 'rate of exponential growth'
        target_velocity = factor
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        
        # Cap the step size to 10% growth per tick to ensure stability
        step_change = np.clip(self.velocity * dt, -0.1, 0.1)
        
        # Apply update: val = val * (1 + change)
        try:
            self.val *= (1.0 + step_change)
        except OverflowError:
            self.val = float('inf')

        # Safety floor
        if self.val < 1e-100: self.val = 1e-100

# ==========================================
# 2. LOG-SCALE MATH ENGINE (Extended for Subset Sum)
# ==========================================

class AstroPhysicsSolver:
    def __init__(self):
        self.variables = {}
        
    def create_var(self, name, rough_magnitude):
        self.variables[name] = AstroDomain(name, initial_scale=rough_magnitude)
    
    def _find_integer_factors(self, target_int, approx_x, approx_y, search_radius=10000000000):
        """
        For integer targets and simple x * y = N equations, find exact integer factors
        close to the approximate floating-point solutions, prioritizing balance.
        Assumes two variables x and y, with x <= y.
        """
        if target_int <= 0:
            return None
        
        # Ensure approx_x <= approx_y for consistency
        if approx_x > approx_y:
            approx_x, approx_y = approx_y, approx_x
        
        best_pair = None
        min_diff = float('inf')
        
        sqrt_n = math.isqrt(target_int)
        
        # Local search near approximations for balanced factors (downward for priority)
        start = max(1, int(approx_x) - search_radius)
        end = min(int(approx_x) + search_radius, sqrt_n)
        
        for cand_x in range(end, start - 1, -1):
            if target_int % cand_x == 0:
                cand_y = target_int // cand_x
                if cand_x <= cand_y:
                    diff = abs(cand_x - approx_x) + abs(cand_y - approx_y)
                    if diff < min_diff:
                        min_diff = diff
                        best_pair = (cand_x, cand_y)
        
        # Fallback: Limited downward for balanced, then small trial for any pair
        if best_pair is None:
            max_checks = 10000000000
            checks = 0
            for i in range(sqrt_n, max(sqrt_n - max_checks, 1), -1):
                if target_int % i == 0:
                    j = target_int // i
                    if i <= j:
                        best_pair = (i, j)
                    break
                checks += 1
            
            if best_pair is None:
                # Small trial for factor
                small_max = 10000000000
                small = None
                for i in range(2, min(small_max + 1, sqrt_n + 1)):
                    if target_int % i == 0:
                        small = i
                        break
                if small:
                    j = target_int // small
                    best_pair = (min(small, j), max(small, j))
                else:
                    best_pair = (1, target_int)
        
        return best_pair
    
    def _solve_subset_sum_exact(self, numbers, target):
        """
        Exact dynamic programming for subset sum (O(n*target), feasible for small target/n).
        Returns subset list or None if impossible.
        """
        if target == 0:
            return []
        n = len(numbers)
        if n == 0 or target < 0:
            return None
        dp = [False] * (target + 1)
        dp[0] = True
        prev = [-1] * (target + 1)
        for num in numbers:
            for s in range(target, num - 1, -1):
                if not dp[s] and dp[s - num]:
                    dp[s] = True
                    prev[s] = s - num
        if not dp[target]:
            return None
        # Reconstruct subset
        subset = []
        s = target
        while s > 0:
            prev_s = prev[s]
            if prev_s == -1:
                break  # Error
            num = s - prev_s
            subset.append(num)
            s = prev_s
        return subset
    
    def _solve_subset_sum_annealing(self, numbers, target, steps=1000000):
        """
        Annealing heuristic: Treat each number as a continuous [0,1] inclusion probability.
        Optimize sum(inclusion_i * numbers_i) to target; threshold to binary {0,1} post-annealing.
        Error = |current_sum - target|; forces adjust inclusions.
        """
        n = len(numbers)
        # Initialize variables as inclusions (0-1 scale)
        for i in range(n):
            var_name = f'incl_{i}'
            self.create_var(var_name, rough_magnitude=0.5)  # Random start ~0.5
        
        for t in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            current_sum = sum(vals[f'incl_{i}'] * numbers[i] for i in range(n))
            
            error = abs(current_sum - target)
            if error < 1e-6:  # Near exact
                break
            
            # Sensitivity: Perturb each inclusion and measure sum change
            perturbation = 0.01  # Small additive for [0,1]
            for i in range(n):
                name = f'incl_{i}'
                domain = self.variables[name]
                orig = domain.val
                clamped_orig = np.clip(orig, 0.0, 1.0)
                
                # Perturb up (if room)
                domain.val = min(1.0, clamped_orig + perturbation)
                sum_new_up = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))
                sens_up = (sum_new_up - current_sum) / perturbation if perturbation > 0 else 0
                
                # Perturb down (if room)
                domain.val = max(0.0, clamped_orig - perturbation)
                sum_new_down = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))
                sens_down = (sum_new_down - current_sum) / (-perturbation) if perturbation > 0 else 0
                
                # Average sensitivity (change per unit inclusion)
                sensitivity = (sens_up + sens_down) / 2.0
                if abs(sensitivity) < 1e-6:
                    sensitivity = numbers[i]  # Default: full weight
                
                # Force: Push towards target (positive if undersum, negative if oversum)
                force = (target - current_sum) / sensitivity if sensitivity != 0 else 0
                force *= 0.1  # Damp for stability
                
                # Update additively (for [0,1]), clamp
                domain.update_multiplicative(force, dt=0.01)  # Reuse, but clamp after
                domain.val = np.clip(domain.val, 0.0, 1.0)
                
                # Restore for next? No, apply sequentially but clamp each time
        
        # Threshold to binary and get subset
        inclusions = {name: round(np.clip(val, 0.0, 1.0)) for name, val in {n: d.val for n, d in self.variables.items()}.items() if name.startswith('incl_')}
        subset = [numbers[i] for i in range(n) if inclusions[f'incl_{i}'] == 1]
        approx_sum = sum(subset)
        return subset if approx_sum == target else None  # Only if exact
    
    def solve(self, equation, steps=1000000000, prefer_integers=False, subset_numbers=None, subset_target=None):
        """
        Solve multiplication equation or subset sum.
        - For multiplication: As before ("x * y = N").
        - For subset sum: Pass subset_numbers=list, subset_target=int. Uses annealing + exact DP fallback.
        Returns {'subset': [nums]} or factors dict.
        """
        if subset_numbers is not None and subset_target is not None:
            # Subset Sum Mode
            print(f"\n[Subset Sum Mode] Numbers: {subset_numbers}, Target: {subset_target}")
            print(f"[System] Set size: {len(subset_numbers)}, Target magnitude: {subset_target}")
            
            # Try exact DP first (if feasible, e.g., target < 10^5)
            if subset_target <= 10000000000 and len(subset_numbers) <= 10000000000:
                exact_subset = self._solve_subset_sum_exact(subset_numbers, subset_target)
                if exact_subset:
                    print(f"[Exact Solution] Subset: {sorted(exact_subset)} (sum: {sum(exact_subset)})")
                    return {'subset': sorted(exact_subset), 'method': 'exact_dp'}
                print("[Exact DP] No solution found.")
            
            # Annealing heuristic
            anneal_subset = self._solve_subset_sum_annealing(subset_numbers, subset_target, steps)
            if anneal_subset:
                print(f"[Annealing Solution] Subset: {sorted(anneal_subset)} (sum: {sum(anneal_subset)})")
                return {'subset': sorted(anneal_subset), 'method': 'annealing'}
            
            print("[Subset Sum] No solution found (NP-hard for large instances).")
            return {'subset': None, 'method': 'failed'}
        
        # Original Multiplication Mode
        print(f"\n[Physics Engine] Target Equation: {equation}")
        
        lhs_str, rhs_str = equation.split('=')
        
        # 1. Parse Target Safely - Handle large integers exactly
        target_int = None
        target_val = None
        try:
            rhs_stripped = rhs_str.strip()
            if 'e' in rhs_stripped.lower() or '.' in rhs_stripped:
                # Scientific or float
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            else:
                # Assume integer literal
                target_int = int(rhs_stripped)
                target_val = float(target_int)
        except (ValueError, OverflowError):
            try:
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            except:
                target_val = float('inf')
                target_int = None
            print("Warning: Target parsing issues, using approximate.")
        
        if target_val is None or target_val == float('inf'):
            print("[System] Target too large or invalid. Stopping.")
            return {}

        # Work in Log10 Space
        if target_val > 0:
            log_target = math.log10(target_val)
        else:
            log_target = -100
            
        print(f"[System] Target Magnitude: 10^{log_target:.2f}")
        
        if target_int is not None:
            print(f"[System] Target is exact integer: {target_int}")
        else:
            print(f"[System] Target approximate: {target_val}")
            
        # 2. Initialize Variables
        import re
        tokens = set(re.findall(r'[a-zA-Z_]+', lhs_str))
        num_vars = len(tokens) if len(tokens) > 0 else 1
        
        # Guess start magnitude (e.g. target 10^300, vars should be 10^100)
        estimated_scale = 10 ** (log_target / num_vars)
        
        for t in tokens: 
            if t not in self.variables: 
                self.create_var(t, rough_magnitude=estimated_scale)
            
        # 3. Annealing Loop
        for t in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            
            # Evaluate LHS
            try:
                current_lhs = eval(lhs_str, {}, vals)
            except OverflowError:
                current_lhs = float('inf')
            
            # Current Log Magnitude
            if current_lhs <= 0: current_lhs = 1e-100
            try:
                log_current = math.log10(current_lhs)
            except ValueError:
                log_current = -100
                
            # ERROR: Difference in Orders of Magnitude
            error = log_current - log_target
            
            # Exit condition (Precision to 8 decimal places of exponent)
            if abs(error) < 1e-8:
                break
                
            # 4. Sensitivity Analysis (Gradient Free)
            # We determine power law relationship without exploding math.
            # How much does LHS change if input changes by 0.1%?
            perturbation = 1.001 # 0.1% change
            log_perturb_delta = math.log10(perturbation) # Constant small number
            
            for name in tokens:
                domain = self.variables[name]
                orig = domain.val
                
                # Test perturbation
                domain.val = orig * perturbation
                
                vals_new = {n: v.val for n, v in self.variables.items()}
                try:
                    lhs_new = eval(lhs_str, {}, vals_new)
                    if lhs_new <= 0: lhs_new = 1e-100
                    log_new = math.log10(lhs_new)
                except:
                    log_new = log_current # No change detected
                
                # Calculate Power Sensitivity (Slope in Log-Log space)
                # E.g., for x^2, sensitivity is 2. For x^3, it is 3.
                sensitivity = (log_new - log_current) / log_perturb_delta
                
                # Restore Value
                domain.val = orig
                
                # 5. Apply Force (Multiplicative)
                # If error is + (too high), we shrink. If - (too low), we grow.
                # We divide by sensitivity: x^3 needs smaller adjustments than x^1.
                if abs(sensitivity) < 0.001: sensitivity = 1.0 # Avoid div by zero
                
                force = -error / sensitivity 
                
                # Scale force for simulation stability
                force *= 10.0 
                
                domain.update_multiplicative(force, dt=0.01)
        
        # Get floating-point results
        float_res = {n: d.val for n, d in self.variables.items()}
        
        # If preferring integers and simple x * y = N
        if prefer_integers and target_int is not None and len(tokens) == 2:
            # Assume tokens are x and y
            if 'x' in tokens and 'y' in tokens:
                approx_x = float_res['x']
                approx_y = float_res['y']
                int_pair = self._find_integer_factors(target_int, approx_x, approx_y)
                if int_pair:
                    # Assign based on approximations for consistency, but prefer larger for x to avoid low
                    pair_small, pair_large = min(int_pair), max(int_pair)
                    if pair_small == approx_x or abs(pair_small - approx_x) < abs(pair_large - approx_x):
                        float_res['x'] = pair_small
                        float_res['y'] = pair_large
                    else:
                        float_res['x'] = pair_large
                        float_res['y'] = pair_small
                    print(f"[Integer Mode] Found factors: {pair_small} * {pair_large} = {target_int} (ratio ~{pair_large / pair_small:.2e}, assigned larger to x if possible)")
        
        return float_res

# ==========================================
# 3. ASTRONOMICAL DEMONSTRATION
# ==========================================

if __name__ == "__main__":
    solver = AstroPhysicsSolver()
    # Randomize Subset Sum at Beginning (your numbers list)
    subset_size = 590
    random_numbers = [random.randint(1, 11500) for _ in range(subset_size)]
    random_subset = random.sample(random_numbers, 8)
    target = sum(random_subset)
    print(f"\n[Randomized Demo] Generated Subset: {sorted(random_subset)}")
    print(f"Generated Target Sum: {target}")

    # Solve the Randomized Subset Sum
    res_subset = solver.solve("", subset_numbers=random_numbers, subset_target=target)
    if res_subset['subset'] is not None:
        print(f"\nSubset Sum Solution: {res_subset['subset']} (method: {res_subset['method']}, sum: {sum(res_subset['subset'])})")
    else:
        print("No subset sum solution found (unlikely, since generated from set).")


