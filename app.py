from flask import Flask, request, render_template, request, redirect, url_for
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenize
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
from flask import send_from_directory


# # flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['OUTPUT_FOLDER'] = './static/outputs/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# # Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
batch_size = 128

# # Load the model and preprocess function
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# Load image embeddings
df = pd.read_pickle('image_embeddings.pickle')
embeddings = np.stack(df['embedding'].values)  # Extract embeddings for PCA or other uses

# PCA setup
pca_components = 50  # Default number of principal components
pca = PCA(n_components=pca_components)
pca_embeddings = pca.fit_transform(embeddings)  # Generate PCA-transformed embeddings
print(f"PCA embeddings shape: {pca_embeddings.shape}")


# # DataFrame to store results
results = []


# #  Helpers
def load_image(file_path):
    """Load and preprocess an image."""
    try:
        image = preprocess_val(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
        return F.normalize(model.encode_image(image), dim=-1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def search_top_k_similar(query_embedding, embeddings, top_k=5):
    """Search for the top K similar items."""
    print(f"Shape of query_embedding: {query_embedding.shape}")
    print(f"Shape of embeddings: {torch.tensor(embeddings).shape}")

    # Compute cosine similarity correctly
    cos_sim = F.cosine_similarity(
        torch.tensor(embeddings).to(device),  # (82783, 512)
        query_embedding,                     # (1, 512)
        dim=1                                # Compare along feature axis
    )

    print(f"Shape of cos_sim: {cos_sim.shape}")
    top_k_indices = torch.topk(cos_sim, top_k).indices.cpu().numpy()
    return df.iloc[top_k_indices], cos_sim[top_k_indices].tolist()



@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    if request.method == "POST":
        query_type = request.form.get("query_type")
        embeddings = np.stack(df['embedding'].values)
        top_k = 5

        try:
            if query_type == "text":
                text_query = request.form.get("text_query")
                if not text_query:
                    error = "Text query is empty"
                else:
                    tokenized_text = tokenize([text_query])
                    text_embedding = F.normalize(model.encode_text(tokenized_text.to(device)), dim=-1)
                    results, similarities = search_top_k_similar(text_embedding, embeddings, top_k)

            elif query_type == "image":
                image_file = request.files.get("image_file")
                if not image_file:
                    error = "No image uploaded"
                else:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                    image_file.save(file_path)
                    image_embedding = load_image(file_path)
                    if image_embedding is None:
                        error = "Failed to process image"
                    else:
                        results, similarities = search_top_k_similar(image_embedding, embeddings, top_k)

            elif query_type == "hybrid":
                text_query = request.form.get("text_query")
                image_file = request.files.get("image_file")
                lam = float(request.form.get("weight", 0.5))

                if not text_query or not image_file:
                    error = "Both image and text are required for hybrid query"
                else:
                    tokenized_text = tokenize([text_query])
                    text_embedding = F.normalize(model.encode_text(tokenized_text.to(device)), dim=-1)

                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                    image_file.save(file_path)
                    image_embedding = load_image(file_path)
                    if image_embedding is None:
                        error = "Failed to process image"
                    else:
                        hybrid_query = F.normalize(lam * text_embedding + (1 - lam) * image_embedding, dim=-1)
                        results, similarities = search_top_k_similar(hybrid_query, embeddings, top_k)
            if results is not None and not error:
                result_images = [
                    {"file_name": row.file_name, "similarity": sim}
                    for (_, row), sim in zip(results.iterrows(), similarities)
                ]
                return render_template("index.html", results=result_images)

        except Exception as e:
            error = str(e)

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)