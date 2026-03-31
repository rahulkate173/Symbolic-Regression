import json
from collections import Counter
from pathlib import Path


def extract_subtrees(tokens, max_len=12):
    """
    Given a flat prefix‑token list like:
        ['add', 'id(x)', 'const(⟨C⟩)']

    returns all subtrees (tuples) of size 4 <= size <= max_len,
    recursively traversing the symbolic tree.
    """
    subtrees = []

    def helper(idx):
        """
        idx: current token index.
        Returns: (size, None)
        """
        if idx >= len(tokens):
            return 0, None

        token = tokens[idx]

        # Binary ops: add, sub, mul, div, pow
        if token in {"add", "sub", "mul", "div", "pow"}:
            l_size, _ = helper(idx + 1)
            r_size, _ = helper(idx + 1 + l_size)
            total_size = 1 + l_size + r_size

        # Unary ops: sin, cos, log, exp, tanh
        elif token in {"sin", "cos", "log", "exp", "tanh"}:
            arg_size, _ = helper(idx + 1)
            total_size = 1 + arg_size

        else:
            # Leaf: id(...), const(), ⟨C⟩, etc.
            total_size = 1

        # Record subtree iff size is in the allowed range
        if 4 <= total_size <= max_len:
            subtree = tuple(tokens[idx : idx + total_size])
            subtrees.append(subtree)

        return total_size, None

    helper(0)
    return subtrees


def mine_subtrees_from_json(json_path, max_len=12, top_k=20):
    """
    Load tokenized trees from JSON and mine the most common subtrees.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}
    tokenized_trees = data["tokenized_trees"]

    counter = Counter()

    for id_tree in tokenized_trees:
        # Remove EOS if present
        if id_tree and id_tree[-1] == vocab.get("EOS", -1):
            id_tree = id_tree[:-1]

        # Convert IDs to tokens
        tokens = [id_to_token.get(i, f"<UNK_{i}>") for i in id_tree]

        # Extract subtrees
        trees = extract_subtrees(tokens, max_len=max_len)
        counter.update(trees)

    # Report top patterns
    print(f"Most common subtrees (max_len={max_len}, top {top_k}):")
    for subtree, cnt in counter.most_common(top_k):
        print(f"{cnt:>4d}: {' '.join(subtree)}")

    return counter


if __name__ == "__main__":
    # Adjust these paths and params to your data
    json_path = "tokenized_gpt_labels_edit.json"  # change if needed
    max_len = 12
    top_k = 20

    mine_subtrees_from_json(json_path, max_len=max_len, top_k=top_k)