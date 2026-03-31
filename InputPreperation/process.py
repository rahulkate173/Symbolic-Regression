import csv
import json
from pathlib import Path

CSV_PATH = "FeynmanEquations.csv"
OUTPUT_JSON = "feynman_equations.json"
CONSTANTS_JSON = "all_constants.json"  # {"pi": 3.14159...}


def extract_identifiers_from_formula(formula_str):
    import re
    # Matches things like: theta, x1, pi, g, etc.
    matches = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula_str)
    return set(matches)

def main():
    # Load constants from JSON once
    with open(CONSTANTS_JSON, "r", encoding="utf-8") as f:
        const_values = json.load(f)  # dict: name -> float

    rows = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            formula = row["Formula"].strip()

            # 1. Build variables dict
            variables = {}
            declared_names = set()

            for i in range(1, 11):  # v1_name ... v10_name
                v_name_col = f"v{i}_name"
                v_low_col = f"v{i}_low"
                v_high_col = f"v{i}_high"

                name = row.get(v_name_col, "").strip()
                if not name:
                    continue

                try:
                    low = float(row[v_low_col])
                    high = float(row[v_high_col])
                except (ValueError, KeyError):
                    low, high = 0.0, 1.0  # fallback

                variables[name] = {"type": "variable", "low": low, "high": high}
                declared_names.add(name)

            # 2. Extract all identifiers from formula
            all_identifiers = extract_identifiers_from_formula(formula)

            # 3. Constants = identifiers that are not variables AND for which we have a value
            constants = {}
            for ident in all_identifiers:
                if ident in declared_names:
                    continue  # it's a variable
                if ident in const_values:
                    constants[ident] = {"type": "constant","value": const_values[ident]}

            rows.append({
                "row": row_idx,
                "original_formula": formula,
                "variables": variables,
                "constants": constants
            })

    # Write final JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Written {len(rows)} rows to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

