from flask import Flask, request, render_template
import os
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load data and model
df = pd.read_pickle('image_embeddings.pickle')  # Ensure this file exists in your directory
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer()

# Helper functions
def load_image(file_path):
    """Load and preprocess an image."""
    try:
        image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
        return F.normalize(model.encode_image(image), dim=-1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def search_embeddings(query_embedding, embeddings, top_k=5):
    """Find top-k most similar images."""
    cos_sim = F.cosine_similarity(torch.tensor(embeddings).to(device), query_embedding.unsqueeze(0))
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

                tokenized_text = tokenizer([text_query])
                text_embedding = F.normalize(model.encode_text(tokenized_text.to(device)), dim=-1)
                results, similarities = search_embeddings(text_embedding, embeddings, top_k)

            elif query_type == "image":
                image_file = request.files.get("image_file")
                if not image_file:
                    error = "No image uploaded"

                file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(file_path)
                image_embedding = load_image(file_path)
                if image_embedding is None:
                    error = "Failed to process image"
                else:
                    results, similarities = search_embeddings(image_embedding, embeddings, top_k)

            elif query_type == "hybrid":
                text_query = request.form.get("text_query")
                image_file = request.files.get("image_file")
                lam = float(request.form.get("weight", 0.5))

                if not text_query or not image_file:
                    error = "Both image and text are required for hybrid query"

                tokenized_text = tokenizer([text_query])
                text_embedding = F.normalize(model.encode_text(tokenized_text.to(device)), dim=-1)

                file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(file_path)
                image_embedding = load_image(file_path)
                if image_embedding is None:
                    error = "Failed to process image"
                else:
                    hybrid_query = F.normalize(lam * text_embedding + (1 - lam) * image_embedding, dim=-1)
                    results, similarities = search_embeddings(hybrid_query, embeddings, top_k)

            if results and not error:
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

