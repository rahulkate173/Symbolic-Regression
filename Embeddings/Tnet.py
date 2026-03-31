import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def determine_max_D(data):
    max_D = 0
    for entry in data:
        if entry["data"]:
            current_D = len(entry["data"][0])
            if current_D > max_D:
                max_D = current_D
    return max_D


def normalize_and_pad_data(cloud, target_D):
    cloud = np.array(cloud)
    if cloud.size == 0 or cloud.shape[0] == 0:
        print("Warning: Empty cloud detected during normalization and padding.")
        return np.array([], dtype=np.float32).reshape(0, target_D)

    current_D = cloud.shape[1]

    if current_D < target_D:
        padding_needed = target_D - current_D
        cloud = np.pad(
            cloud, ((0, 0), (0, padding_needed)), "constant", constant_values=0
        )
    elif current_D > target_D:
        print(f"Warning: Truncating features from {current_D} to {target_D}.")
        cloud = cloud[:, :target_D]

    mean = cloud.mean(axis=0)
    std = cloud.std(axis=0)
    std[std == 0] = 1e-8

    normalized = (cloud - mean) / std
    return normalized.astype(np.float32)


class TNet(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, embed_dim)
        self.mlp2 = nn.Linear(embed_dim, 2 * embed_dim)
        self.mlp3 = nn.Linear(2 * embed_dim, 4 * embed_dim)
        self.final = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x, _ = torch.max(x, dim=1)  # max pooling over points
        return self.final(x)


def main(input_json, output_json, embed_dim=128):
    data = load_data(input_json)
    print(f"Loaded {len(data)} equations")

    max_D = determine_max_D(data)
    print(f"Determined maximum feature dimension (D) across all clouds: {max_D}")

    tnet = TNet(input_dim=max_D, embed_dim=embed_dim)
    print(
        f"(Initialized TNet with input_dim={max_D} and embed_dim={embed_dim})"
    )

    embeddings = []

    for entry in data:
        cloud_raw = entry["data"]
        cloud = normalize_and_pad_data(cloud_raw, max_D)

        if cloud.shape[0] == 0:
            print(
                f"Skipping row_id {entry['row_id']} (empty cloud after processing)."
            )
            continue

        cloud_tensor = torch.tensor(cloud).unsqueeze(0)  # [1, n, d]

        with torch.no_grad():
            emb = tnet(cloud_tensor).squeeze().numpy()
            embeddings.append(
                {"row_id": entry["row_id"], "embedding": emb.tolist()}
            )

    # Save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Saved to {output_json}")


# === Use this in your code ===
if __name__ == "__main__":
    input_json = "cloudPoints.json"       # change to your path
    output_json = "tnet_embeddings.json"  # change to your path
    main(input_json, output_json, embed_dim=128)