import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

#  Setup 
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# Step 1: Text Embedding 
texts = [
    "a peaceful place to be alone",
    "a crowded and noisy street",
    "a quiet reading corner",
    "a place of worship or meditation",
    "a vibrant public market"
]

with torch.no_grad():
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    text_embeddings = model.get_text_features(**inputs)

# === Step 1: Image Embedding ===
image_folder = "Path_to_your_Image_folder"
image_embeddings = []
image_paths = []

with torch.no_grad():
    for file in tqdm(os.listdir(image_folder), desc="Processing Images"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_folder, file)
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            image_embedding = model.get_image_features(**inputs)

            image_embeddings.append(image_embedding)
            image_paths.append(path)

image_embeddings = torch.cat(image_embeddings, dim=0)

print("\nText embeddings shape:", text_embeddings.shape)
print("Image embeddings shape:", image_embeddings.shape)
print("Processed image files:", image_paths)

# Optional: Save Embeddings 
os.makedirs("text_embeddings", exist_ok=True)
for i, text in enumerate(texts):
    filename = f"text_embeddings/text_{i+1}.txt"
    np.savetxt(filename, text_embeddings[i].cpu().numpy(), fmt="%.6f")

os.makedirs("image_embeddings", exist_ok=True)
for i, path in enumerate(image_paths):
    name = os.path.splitext(os.path.basename(path))[0]
    filename = f"image_embeddings/{name}.txt"
    np.savetxt(filename, image_embeddings[i].cpu().numpy(), fmt="%.6f")

# Step 2: Normalize 
text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

# Step 3: Similarity Matrix 
similarity = text_embeddings @ image_embeddings.T
top_indices = similarity.argmax(dim=1)

# Step 4: Print Matching Results 
print("\nTop Matches:")
for i, idx in enumerate(top_indices):
    print(f"Text {i+1}: \"{texts[i]}\" → Image: {os.path.basename(image_paths[idx])}")

# Step 5: Visualize Matches
cols = 2
rows = (len(texts) + cols - 1) // cols
plt.figure(figsize=(12, 5 * rows))

for i, idx in enumerate(top_indices):
    image_path = image_paths[idx]
    image = Image.open(image_path)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Text {i+1}:\n"{texts[i]}"', fontsize=10)

plt.tight_layout()
plt.show()
