from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

DATASET_PATH = "faces_datasets"
THRESHOLD = 0.6

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Step 1: Load dataset

print("Loading dataset...")

dataset_embeddings = {}

for user in os.listdir(DATASET_PATH):
    user_path = os.path.join(DATASET_PATH, user)

    if os.path.isdir(user_path):
        embeddings = []

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)

            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(embedding)

            except:
                print("Error processing:", img_path)

        if embeddings:
            dataset_embeddings[user] = embeddings

print("Dataset loaded:", list(dataset_embeddings.keys()))

# Step 2: Recognition

def recognize(image_path):

    try:
        test_embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        best_user = None
        best_score = -1

        for user, embeddings in dataset_embeddings.items():
            for emb in embeddings:
                score = cosine_similarity(test_embedding, emb)

                if score > best_score:
                    best_score = score
                    best_user = user

        if best_score >= THRESHOLD:
            similarity_percent = best_score * 100
            return True, best_user, similarity_percent
        else:
            return False, "Unknown", best_score * 100

    except Exception as e:
        print(e)
        return False, "Error", 0

# Flask Routes

@app.route("/")
def index():
    return render_template("index.htm")


@app.route("/verify", methods=["POST"])
def verify():
    file = request.files["image"]
    path = "temp.jpg"
    file.save(path)

    ok, name, sim = recognize(path)

    if ok:
        return jsonify({
            "status": "success",
            "name": name,
            "similarity": round(sim, 2)
        })
    else:
        return jsonify({
            "status": "fail",
            "similarity": round(sim, 2)
        })

# Run

if __name__ == "__main__":
    app.run(debug=True)