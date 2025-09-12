import os

# Paths
cloth_dir = "data/test/cloth"
image_dir = "data/test/image"
agnostic_dir = "data/test/agnostic-v3.2"

# Output file
output_file = "test_pairs_clean.txt"

# Collect available files
cloth_files = set(os.listdir(cloth_dir))
image_files = set(os.listdir(image_dir))
agnostic_files = set(os.listdir(agnostic_dir))

valid_pairs = []

for cloth in cloth_files:
    if "_0" not in cloth:
        continue  # skip if not a cloth file
    base = cloth.replace("_0.jpg", "")
    person = base + "_1.jpg"

    if person in image_files and person in agnostic_files:
        valid_pairs.append(f"{cloth} {person}")

# Save clean pairs list
with open(output_file, "w") as f:
    f.write("\n".join(valid_pairs))

print(f"✅ Generated {output_file} with {len(valid_pairs)} valid pairs.")
