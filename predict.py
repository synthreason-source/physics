import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

def generate_expanded_framework():
    print("Fetching standard element data from PubChem...")
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV/?response_type=save&response_basename=PubChemElements_all"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"\nConnection Error: {e}")
        return

    # Clean and rename columns
    df = df[['AtomicNumber', 'Symbol', 'Name', 'AtomicMass', 'Electronegativity', 'AtomicRadius', 'IonizationEnergy']].copy()
    df.columns = ['Z', 'Symbol', 'Element', 'A', 'EN', 'Rad_pm', 'IE_eV']

    # Filter up to Uranium (Z=92) and drop missing data
    df = df[df['Z'] <= 92]
    df = df.dropna(subset=['A', 'Rad_pm', 'IE_eV']).copy()

    # Calculate Binding Energy (Barrier Density)
    def calculate_be_per_nucleon(Z, A_float):
        A = round(A_float)
        if A == 0 or Z == 0: return 0
        N = A - Z
        a_v, a_s, a_c, a_a, a_p = 15.8, 18.3, 0.714, 23.2, 12.0
        vol = a_v * A
        surf = a_s * (A**(2/3))
        coul = a_c * Z * (Z - 1) / (A**(1/3)) if A > 0 else 0
        asym = a_a * ((A - 2*Z)**2) / A if A > 0 else 0
        
        if Z % 2 == 0 and N % 2 == 0: pair = a_p / (A**0.5)
        elif Z % 2 != 0 and N % 2 != 0: pair = -a_p / (A**0.5)
        else: pair = 0
            
        total_be = vol - surf - coul - asym + pair
        return max(total_be / A, 0)

    print("Calculating Theoretical Gravity Barrier Properties...")
    df['Barrier_Density'] = df.apply(lambda row: calculate_be_per_nucleon(row['Z'], row['A']), axis=1)
    
    # 1. Core Framework Properties
    df['Barrier_Thickness'] = df['A'] * df['Barrier_Density']
    df['Lorentz_Potential'] = 1000 / df['Barrier_Thickness']
    df['Barrier_Stress'] = (df['Z']**2) / (df['A']**(1/3))

    # 2. Electrical Conductivity Prediction
    df['Predicted_Conductivity'] = df['Barrier_Thickness'] / df['IE_eV']

    # 3. Photoemission Threshold Prediction
    df['Predicted_Photoemission'] = df['Lorentz_Potential'] * df['Barrier_Density']

    # 4. Magnetic Resonance Index
    max_density = df['Barrier_Density'].max()
    df['Magnetic_Index'] = np.exp(df['Barrier_Density'] - max_density) * df['Barrier_Thickness'] / df['Rad_pm']

    # Save to CSV
    output_file = "comprehensive_theoretical_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Success! Dataset saved to '{output_file}'\n")

    # --- NEW: PRINT ALL CALCULATED DETAILS ---
    # Configure pandas to print the full table beautifully without truncation
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    # Select specific columns to print so it fits horizontally on a screen
    print_cols = ['Z', 'Element', 'Barrier_Density', 'Barrier_Thickness', 
                  'Lorentz_Potential', 'Barrier_Stress', 
                  'Predicted_Conductivity', 'Predicted_Photoemission', 'Magnetic_Index']
    
    print("="*125)
    print("COMPLETE THEORETICAL PREDICTIONS FOR ALL ELEMENTS")
    print("="*125)
    print(df[print_cols].to_string(index=False))
    print("="*125)

    # Plotting the 6-Panel Dashboard
    print("\nGenerating visual dashboard...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Barrier Nuclear Barrier Predictions vs Actual Properties", fontsize=18, fontweight='bold')

    key_elements = ['He', 'C', 'O', 'Fe', 'Ag', 'Au', 'U']
    def annotate_points(ax, x_col, y_col):
        for idx, row in df.iterrows():
            if row['Symbol'] in key_elements:
                ax.annotate(row['Symbol'], (row[x_col], row[y_col]), fontsize=9, alpha=0.8)

    axs[0, 0].scatter(df['Barrier_Thickness'], df['Rad_pm'], c='blue', alpha=0.6)
    annotate_points(axs[0, 0], 'Barrier_Thickness', 'Rad_pm')
    axs[0, 0].set_title("1. Atomic Radius vs Barrier Thickness")
    axs[0, 0].set_xlabel("Barrier Thickness")
    axs[0, 0].set_ylabel("Actual Radius (pm)")

    axs[0, 1].scatter(df['Lorentz_Potential'], df['IE_eV'], c='red', alpha=0.6)
    annotate_points(axs[0, 1], 'Lorentz_Potential', 'IE_eV')
    axs[0, 1].set_title("2. Ionization Energy vs Lorentz Potential")
    axs[0, 1].set_xlabel("Lorentz Potential (1/Thickness)")
    axs[0, 1].set_ylabel("Actual IE (eV)")

    axs[0, 2].plot(df['Z'], df['Barrier_Stress'], marker='o', c='purple', linestyle='-', alpha=0.6)
    annotate_points(axs[0, 2], 'Z', 'Barrier_Stress')
    axs[0, 2].set_title("3. Barrier Stress vs Atomic Number")
    axs[0, 2].set_xlabel("Atomic Number (Z)")
    axs[0, 2].set_ylabel("Barrier Stress (Instability)")

    axs[1, 0].scatter(df['Z'], df['Predicted_Conductivity'], c='green', alpha=0.6)
    annotate_points(axs[1, 0], 'Z', 'Predicted_Conductivity')
    axs[1, 0].set_title("4. Predicted Electrical Conductivity")
    axs[1, 0].set_xlabel("Atomic Number (Z)")
    axs[1, 0].set_ylabel("Conductivity Index (Thickness / IE)")

    axs[1, 1].scatter(df['Lorentz_Potential'], df['Predicted_Photoemission'], c='orange', alpha=0.6)
    annotate_points(axs[1, 1], 'Lorentz_Potential', 'Predicted_Photoemission')
    axs[1, 1].set_title("5. Photoemission Threshold Prediction")
    axs[1, 1].set_xlabel("Lorentz Potential")
    axs[1, 1].set_ylabel("Photoemission Index")

    axs[1, 2].scatter(df['Z'], df['Magnetic_Index'], c='darkcyan', alpha=0.6)
    annotate_points(axs[1, 2], 'Z', 'Magnetic_Index')
    axs[1, 2].set_title("6. Predicted Magnetic Index")
    axs[1, 2].set_xlabel("Atomic Number (Z)")
    axs[1, 2].set_ylabel("Magnetic Index (Based on Barrier Density)")

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    generate_expanded_framework()
