import MDAnalysis as mda
import numpy as np
import pandas as pd
import sys

TOPOLOGY_FILE = sys.argv[1]
TRAJECTORY_FILE = sys.argv[2]
MICELLE_SELECTION = 'resname CTA' 
OUTPUT_CSV_FILE = 'micelle_shape_analysis.csv'
-
def get_micelle_gyration_tensor_eigenvalues(atom_group):
    """
    Calculates the eigenvalues of the mass-weighted gyration tensor for an AtomGroup.
    Orders them as lambda1 >= lambda2 >= lambda3.
    """
    com = atom_group.center_of_mass()
    coordinates = atom_group.positions - com
    masses = atom_group.masses
    total_mass = atom_group.total_mass()

    Rg_tensor = np.zeros((3, 3))
    for i in range(3): 
        for j in range(3): 
            Rg_tensor[i, j] = np.sum(masses * coordinates[:, i] * coordinates[:, j]) / total_mass

    eigenvalues = np.linalg.eigvalsh(Rg_tensor) 
    eigenvalues_sorted = np.sort(eigenvalues)[::-1] 
    return eigenvalues_sorted 

def calculate_shape_parameters(eigenvalues):
    """
    Calculates Rg, K^2, and the principal radii (R1, R2, R3) from eigenvalues.
    """
    l1, l2, l3 = eigenvalues

    rg_sq = l1 + l2 + l3
    if rg_sq <= 0: 
        return 0, 0, 0, 0, 0, 0

    rg = np.sqrt(rg_sq)
    K_sq = 1.0 - 3.0 * (l1*l2 + l2*l3 + l3*l1) / (rg_sq**2) if rg_sq > 0 else 0

    #safe sqrt avoid floating point issues with 0
    r1 = np.sqrt(max(0, l1))
    r2 = np.sqrt(max(0, l2))
    r3 = np.sqrt(max(0, l3))

  acylindricity = l2 - l3

    return rg, K_sq, r1, r2, r3, acylindricity

print(f"Loading trajectory: {TOPOLOGY_FILE}, {TRAJECTORY_FILE}")
u = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE)
print(f"Trajectory loaded with {len(u.trajectory)} frames.")

micelle_group = u.select_atoms(MICELLE_SELECTION)
results = []

print("Starting frame-by-frame analysis...")
for ts in u.trajectory:
    if ts.frame % 100 == 0: 
        print(f"  Processing frame {ts.frame}...")

    if len(micelle_group) < 3: 
        print(f"  Skipping frame {ts.frame}, not enough atoms: {len(micelle_group)}")
        results.append([ts.frame, ts.time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        continue

    eigenvalues = get_micelle_gyration_tensor_eigenvalues(micelle_group)
    rg, K_sq, r1, r2, r3, acylindricity = calculate_shape_parameters(eigenvalues)

    results.append([ts.frame, ts.time, rg, r1, r2, r3, K_sq, acylindricity])

print("Analysis complete.")

df_results = pd.DataFrame(results, columns=[
    'Frame', 'Time_ps', 'Rg_A', 'R1_A', 'R2_A', 'R3_A', 'K_sq', 'Acylindricity_A2'
])


df_results.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\nResults saved to {OUTPUT_CSV_FILE}")

print("\n" + "="*50)
print(" SUMMARY FOR TABLE SX (Mean ± Standard Deviation)")
print("="*50)

df_clean = df_results.dropna()

metrics = ['Rg_A', 'R1_A', 'R2_A', 'R3_A', 'K_sq']
for col in metrics:
    mean_val = df_clean[col].mean()
    std_val  = df_clean[col].std()
    
    if col == 'K_sq':
        print(f"{col:>6}: {mean_val:.3f} (± {std_val:.3f})")
    else:
        print(f"{col:>6}: {mean_val:.2f} (± {std_val:.2f}) Å")

print("="*50)
