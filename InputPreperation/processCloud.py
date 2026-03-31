import json
import csv
import os
from pathlib import Path

# Paths
FEYNMAN_JSON = "..\feynman_equations.json"     
DATA_FOLDER  = "..\Feynman_with_units"              
OUTPUT_JSON  = "cloudPoints.json"


def main():
    # 1. Load JSON structure (from CSV → JSON)
    with open(FEYNMAN_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)  # list of dicts
    input_specs = {}

    for entry in entries:
        row_id = entry["row"]
        variables = entry.get("variables", {})
        inputs = sorted(variables)  # or use order from CSV if you prefer
        num_inputs = len(inputs)

        input_specs[row_id] = {
            "inputs": inputs,
            "num_inputs": num_inputs
        }

    # 3. For each row_id, read data file and sample 1000 points
    cloud_points = []

    for row_id, spec in input_specs.items():
        row_entry = entries[row_id]
        formula = row_entry["original_formula"]
        filename = f"eq_{row_id}.txt"  

    # Let’s do it properly: read CSV to get filename per row
    row_to_filename = {}
    with open("FeynmanEquations.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            filename = row["Filename"].strip()  # first column name
            row_to_filename[row_idx] = filename

    # Rebuild cloud points using CSV filenames
    cloud_points = []

    for row_id, spec in input_specs.items():
        filename = row_to_filename.get(row_id)
        if filename is None:
            print(f"No filename for row_id={row_id}, skipping")
            continue

        file_path = Path(DATA_FOLDER) / filename
        if not file_path.exists():
            print(f"File not found: {file_path}, skipping row_id={row_id}")
            continue

        inputs_list = spec["inputs"]
        num_inputs = spec["num_inputs"]

        data = []

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split by space or comma
                try:
                    parts = line.replace(",", " ").split()
                    vals = [float(x) for x in parts if x.strip()]
                except ValueError:
                    continue

                # First num_inputs → inputs, last value → output
                if len(vals) < num_inputs + 1:
                    continue

                # inputs: first N, output: last
                input_vals = vals[:num_inputs]
                output_val = vals[-1]
                row_data = input_vals + [output_val]

                data.append(row_data)
                count += 1
                if count >= 100000:
                    break

        # 3.2. Append to cloudPoints
        cloud_points.append({
            "row_id": row_id,
            "inputs": inputs_list,
            "formula": entries[row_id]["original_formula"],
            "data": data
        })

    # 4. Write final cloudPoints.json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cloud_points, f, indent=2)

    print(f"Wrote {len(cloud_points)} rows to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()