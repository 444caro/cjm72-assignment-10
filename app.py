import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# # Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
batch_size = 128
image_folder = ''  # Replace with your folder path

# # Load the model and preprocess function
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# # Image transformations (using preprocess_val from open_clip)
transform = preprocess_val

# # Collect all image paths
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
print('Number of images:', len(image_paths))
# # DataFrame to store results
results = []

# # Function to load and preprocess images
def load_images(batch_paths):
  images = []
  for path in batch_paths:
    try:
      image = Image.open(path).convert("RGB")
      images.append(transform(image))
    except Exception as e:
      print(f"Error loading image {path}: {e}")
  return torch.stack(images) if images else None

# # Process images in batches
with torch.no_grad():
  for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
    batch_paths = image_paths[i:i + batch_size]
    images = load_images(batch_paths)
    if images is None:  # Skip if no valid images in this batch
      continue 
    images = images.to(device)
    embeddings = model.encode_image(images)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings
    for path, emb in zip(batch_paths, embeddings):
      results.append({"file_name": os.path.basename(path), "embedding": emb.cpu().numpy()})

# Save results to a DataFrame
df = pd.DataFrame(results)
df.to_pickle('image_embeddings.pickle')
