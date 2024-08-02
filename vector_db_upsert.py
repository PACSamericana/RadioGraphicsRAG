import os
import json
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# Initialize OpenAI client
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

# Initialize Qdrant Client
QDRANT_MAIN_URL = "[QDRANT_URL]"
QDRANT_API_KEY = "[QDRANT_API_KEY]"
qdrant_client = QdrantClient(url=QDRANT_MAIN_URL, api_key=QDRANT_API_KEY)

# Ensure the collection exists and is configured properly
qdrant_client.create_collection(
    collection_name="[COLLECTION_NAME]",
    vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE,)
)

def generate_embedding(text, model="text-embedding-3-large"):
    try:
        response = client.embeddings.create(input=text, model=model)
        # Convert the response to a dictionary and access the embedding
        embedding = response.dict()['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"An error occurred while generating embedding: {e}")
        return []

folder_path = '[FOLDER_PATH]'
collection_name = "[COLLECTION_NAME]"

# Iterate over JSON files and upsert embeddings one by one
for idx, file_name in enumerate(os.listdir(folder_path)):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)

        embedding = generate_embedding(data['text'])
        if embedding is None:  # Check if embedding generation failed
            print(f"Skipping upsert for {file_name} due to failed embedding generation.")
            continue  # Skip this file

        point_id = idx  # Or derive a unique ID from your data
        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=point_id, vector=embedding, payload={"citation": data['citation'], "text": data['text']})],
            )
            print(f"Successfully upserted: {file_name}")
        except Exception as e:
            print(f"Error upserting {file_name}: {e}")

