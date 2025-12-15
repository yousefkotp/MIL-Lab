#!/usr/bin/env python3
"""
Script to find config.yaml files with sample_col != 'filename'
"""

import yaml
from pathlib import Path

# Hardcoded search path
SEARCH_PATH = "/home/kotpaz/scratch/tasks/custom"

def main():
    search_dir = Path(SEARCH_PATH)

    # Find all config.yaml files recursively
    config_files = search_dir.rglob("config.yaml")

    print(f"Searching for config.yaml files in: {SEARCH_PATH}\n")

    files_with_issues = []

    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Check if sample_col exists and is not 'filename'
            if config and 'sample_col' in config:
                sample_col_value = config['sample_col']

                # Only 'filename' is allowed, anything else should be reported
                if sample_col_value != 'filename':
                    print(f"Found config with sample_col: {sample_col_value}")
                    print(f"  Path: {config_file}")
                    print()
                    files_with_issues.append(config_file)

        except Exception as e:
            print(f"Error reading {config_file}: {e}")
            print()

    # Summary
    print("=" * 70)
    if files_with_issues:
        print(f"\nFound {len(files_with_issues)} config file(s) with sample_col != 'filename':")
        for file_path in files_with_issues:
            print(f"  - {file_path}")
    else:
        print("\nAll config files have sample_col: filename (or no sample_col entry)")

if __name__ == "__main__":
    main()
