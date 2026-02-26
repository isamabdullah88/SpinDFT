import os
import ase.db
from ase.io import write

def persist_relaxed(dbpath, stype='Uniaxial', outdir="RelaxedAtoms"):
    """
    Reads an ASE database and extracts the atomic structures into individual files.
    Assumes each row in the database has a 'strain' key-value pair.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Connect to the ASE database
    print(f"Connecting to database: {dbpath}")
    try:
        db = ase.db.connect(dbpath)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    extracted_count = 0
    
    # Iterate through all rows in the database
    for row in db.select():
        # Fetch the strain value. 
        # (Change 'strain' to 'eps' or whatever key you used when saving to the db)
        strain = row.get('strain_value')
        
        if strain is not None:
            atoms = row.toatoms()
            
            # We use .json extension because ASE natively supports it. 
            # It perfectly preserves fractional coords, cell dimensions, and pbc.
            # Formatting to 3 decimal places (e.g., strain_0.020.json) to avoid float rounding errors in filenames.
            filename = f"Strain_{stype}_X_{strain:.4f}.json"
            filepath = os.path.join(outdir, filename)
            
            # Write the atoms object to the file
            write(filepath, atoms)
            print(f"Extracted strain {strain:.3f} -> Saved to {filepath}")
            extracted_count += 1
        else:
            print(f"Warning: Row {row.id} does not have a 'strain' key. Skipping.")

    print(f"\nExtraction complete. {extracted_count} structures saved to '{outdir}/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract relaxed atomic structures from an ASE database.")
    parser.add_argument('--dbpath', type=str, required=True, help='Path to the ASE database file.')
    parser.add_argument('--stype', type=str, default='Uniaxial_X', help='Strain type to include in filenames (default: Uniaxial_X).')
    parser.add_argument('--outdir', type=str, default='RelaxedAtoms', help='Output directory to save the extracted structures (default: RelaxedAtoms).')
    args = parser.parse_args()
    persist_relaxed(args.dbpath, stype=args.stype, outdir=args.outdir)