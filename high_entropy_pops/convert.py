import sys, types, gzip, joblib, pandas as pd
from ase.io import write
import numpy as np

# --- Patch for old pandas pickles (pandas.core.indexes.numeric issue) ---
import pandas.core.indexes
sys.modules['pandas.core.indexes.numeric'] = types.ModuleType('pandas.core.indexes.numeric')
setattr(sys.modules['pandas.core.indexes.numeric'], 'Int64Index', pd.Index)

# --- Load your DataFrame ---
filename = "df_Al_test.pckl.gzip"  # or df_Al_test.pickl if not compressed

try:
    with gzip.open(filename, "rb") as f:
        df = joblib.load(f)
except OSError:
    df = joblib.load(filename)


# print(f"✅ Loaded DataFrame with shape {df.shape}")
# print("Columns:", df.columns.tolist())

# # --- Build ASE Atoms list with energies and forces ---
# atoms_list = []
# for _, row in df.iterrows():
#     atoms = row["ase_atoms"]

#     # attach scalar properties
#     if "energy" in row:
#         atoms.info["dft_energy"] = float(row["energy"])

#     # attach per-atom arrays
#     if "forces" in row:
#         atoms.arrays["dft_forces"] = np.array(row["forces"])
#         atoms.info["dft_forces"]   = np.array(row["forces"])
#     atoms.arrays["numbers"] = 13 * np.ones(len(atoms), dtype=int)
#     atoms_list.append(atoms)

# # --- Save to .extxyz ---
# out_file = "df_test_Al.xyz"
# write("df_test_Al.xyz", atoms_list)  # omit id/type unless needed


# print(f"\n✅ Saved {len(atoms_list)} structures with energy/forces to {out_file}")

def write_custom_extxyz_manual(df, filename, config_type):
    """Write structures in strict .extxyz format with renamed fields and no extras."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            atoms = row["ase_atoms"]
            natoms = len(atoms)
            lattice = " ".join(f"{v:.6f}" for v in atoms.cell.array.flatten())
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            forces = np.array(row["forces"])
            volume = atoms.get_volume()

            # Prefer free_energy over energy
            energy = float(row["energy"])

            # Get stress and compute virial
            #stress = atoms.get_stress(voigt=False)
            #virial_tensor = -stress * volume
            #virial_flat = " ".join(f"{v:.8f}" for v in virial_tensor.flatten())

            #This is I am just going to pass into it
            
            # Header line
            f.write(
                f'{natoms}\n'
                f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:dft_forces:R:3 '
                f'dft_energy={energy:.10f} '
                f'config_type={config_type} pbc="T T T"\n'
            )
            # --- MODIFICATION END ---
            
            # Atom lines
            for i in range(natoms):
                f.write(
                    f"{symbols[i]:<2}  "
                    f"{positions[i,0]:15.8f}  {positions[i,1]:15.8f}  {positions[i,2]:15.8f}  "
                    f"{forces[i,0]:15.8f}  {forces[i,1]:15.8f}  {forces[i,2]:15.8f}\n"
                )

write_custom_extxyz_manual(df, "manual_df_test_Al.xyz", "high_entropy")