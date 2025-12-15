#!/usr/bin/env python3
"""
Script to extract and combine filename columns from multiple CSV files
"""

import pandas as pd
from pathlib import Path

# Hardcoded list of CSV files to process
CSV_FILES = [
    "/home/kotpaz/scratch/tasks/custom/bracs/coarse/task.csv",
    "/home/kotpaz/scratch/tasks/custom/bracs/fine/task.csv",
    "/home/kotpaz/scratch/tasks/custom/camelyon17/breast_cancer_metastases/task.csv",
    "/home/kotpaz/scratch/tasks/custom/dhmc_kidney/morphological_subtyping/task.csv",
    "/home/kotpaz/scratch/tasks/custom/dhmc_luad/histologic_pattern/task.csv",
    "/home/kotpaz/scratch/tasks/custom/ebrains/diagnosis/task.csv",
    "/home/kotpaz/scratch/tasks/custom/ebrains/diagnosis_group/task.csv",
    "/home/kotpaz/scratch/tasks/custom/ebrains/idh_status/task.csv",
    "/home/kotpaz/scratch/tasks/custom/imp/grade/task.csv",
    "/home/kotpaz/scratch/tasks/custom/panda/prostate_cancer_grade/task.csv",
]

# Output file path
OUTPUT_FILE = "combined_filenames.csv"

def main():
    all_filenames = []

    print(f"Processing {len(CSV_FILES)} CSV files...\n")

    for csv_file in CSV_FILES:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if 'filename' column exists
            if 'filename' not in df.columns:
                print(f"Warning: 'filename' column not found in {csv_file}")
                print(f"  Available columns: {', '.join(df.columns)}")
                continue

            # Extract filename column
            filenames = df['filename'].tolist()
            all_filenames.extend(filenames)

            print(f"✓ {csv_file}")
            print(f"  Found {len(filenames)} filenames")

        except FileNotFoundError:
            print(f"✗ File not found: {csv_file}")
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")

    # Remove duplicates while preserving order
    print(f"\nTotal filenames collected: {len(all_filenames)}")
    unique_filenames = list(dict.fromkeys(all_filenames))
    print(f"Unique filenames: {len(unique_filenames)}")

    # Create output DataFrame
    output_df = pd.DataFrame({'filename': unique_filenames})

    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Combined filenames saved to: {OUTPUT_FILE}")
    print(f"  Total rows: {len(output_df)}")

if __name__ == "__main__":
    main()
