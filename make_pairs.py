import os

# Paths
data_root = "data/test"
image_dir = os.path.join(data_root, "image")
cloth_dir = os.path.join(data_root, "cloth")
pairs_file = "test_pairs.txt"

# Get sorted file lists
images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
cloths = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".jpg")])

# Simple pairing: each person image with first cloth (or same index cloth if lengths match)
pairs = []
if len(images) <= len(cloths):
    for i, img in enumerate(images):
        pairs.append(f"{img} {cloths[i % len(cloths)]}")
else:
    for i, cloth in enumerate(cloths):
        pairs.append(f"{images[i % len(images)]} {cloth}")

# Save pairs.txt
with open(pairs_file, "w") as f:
    f.write("\n".join(pairs))

print(f"✅ {pairs_file} generated with {len(pairs)} pairs!")
