import json
import re

def main(input_json, output_json):
    # === Step 1: Load your JSON ===
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} entries from {input_json}")

    # === Step 2: Mask all constants with ⟨C⟩ ===
    masked_entries = []
    for entry in data:
        tree = entry.get("symbolic_parse_tree")
        if tree is None:
            # If you called it "masked_parse_tree" already, adjust key:
            tree = entry.get("masked_parse_tree")

        if tree is None or not isinstance(tree, str):
            masked_tree = ""
        else:
            # Replace const(...) → const(⟨C⟩)
            masked_tree = re.sub(r"const\(([^)]+)\)", "const(⟨C⟩)", tree)

        masked_entries.append({
            **entry,
            "masked_parse_tree": masked_tree
        })

    # Optional: save masked trees to a file
    masked_out = "masked_parse_trees.json"
    with open(masked_out, "w", encoding="utf-8") as f:
        json.dump(masked_entries, f, indent=2)
    print(f"✅ Masked trees saved to {masked_out}")


    # === Step 3: Parse string trees into nested lists ===
    def str_expr_to_tree(expr_str):
        def tokenize(s):
            tokens = []
            current = ""
            for c in s:
                if c in "(), \t\n":
                    if current:
                        tokens.append(current)
                        current = ""
                    if c not in " \t\n":
                        tokens.append(c)
                else:
                    current += c
            if current:
                tokens.append(current)
            return tokens

        def parse(tokens):
            if not tokens:
                return None

            token = tokens.pop(0)

            if token not in "(),":
                if tokens and tokens[0] == "(":
                    func_name = token
                    tokens.pop(0)  # '('
                    args = []
                    while tokens and tokens[0] != ")":
                        args.append(parse(tokens))
                        if tokens and tokens[0] == ",":
                            tokens.pop(0)
                    if tokens and tokens[0] == ")":
                        tokens.pop(0)  # ')'
                    return [func_name] + args
                else:
                    return token
            elif token == "(":
                expr = parse(tokens)
                if tokens and tokens[0] == ")":
                    tokens.pop(0)
                return expr
            return None

        tokens = tokenize(expr_str)
        return parse(tokens)

    # === Step 4: Flatten to prefix tokens ===
    def flatten_tree(tree):
        if isinstance(tree, str):
            return [tree]
        elif isinstance(tree, list):
            return [tree[0]] + [t for arg in tree[1:] for t in flatten_tree(arg)]
        else:
            raise ValueError("Unexpected tree format")

    tokenized_trees = []
    for entry in masked_entries:
        expr_str = entry.get("masked_parse_tree", "")
        if not expr_str.strip():
            tokenized_trees.append([])
            continue

        tree = str_expr_to_tree(expr_str)
        if tree is None:
            tokenized_trees.append([])
        else:
            tokens = flatten_tree(tree)
            tokenized_trees.append(tokens)

    # === Step 5: Build vocab + add special tokens ===
    vocab = {}
    for tokens in tokenized_trees:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    # Special tokens
    if "<PAD>" not in vocab:
        vocab["<PAD>"] = len(vocab)
    if "<EOS>" not in vocab:
        vocab["<EOS>"] = len(vocab)

    # === Step 6: Convert to ID sequences (with <EOS>) ===
    tokenized_id_seqs = []
    for tokens in tokenized_trees:
        ids = [vocab[tok] for tok in tokens] + [vocab["<EOS>"]]
        tokenized_id_seqs.append(ids)

    # === Step 7: Save result ===
    output = {
        "vocab": vocab,
        "tokenized_trees": tokenized_id_seqs
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Tokenized trees and vocab saved to {output_json}")
    print(f"🔢 Vocabulary size (including <PAD>, <EOS>): {len(vocab)}")


# === How to use this in your code ===
if __name__ == "__main__":
    input_json = "feynman_equations.json"        # your masked JSON
    output_json = "tokenized_gpt_labels_edit.json" # your output
    main(input_json, output_json)