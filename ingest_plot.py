import requests
import pandas as pd
import psycopg2
from transformers import AutoTokenizer
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

class SingletonTokenizer:
    _instance = None

    @staticmethod
    def get_instance(model_name="bert-base-uncased"):
        if SingletonTokenizer._instance is None:
            SingletonTokenizer._instance = AutoTokenizer.from_pretrained(model_name)
        return SingletonTokenizer._instance


class TextEmbedder:
    def __init__(self, api_url, db_config, model_name="bert-base-uncased"):
        self.api_url = api_url
        self.conn = psycopg2.connect(**db_config)
        self.tokenizer = SingletonTokenizer.get_instance(model_name)  # Singleton tokenizer instance

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def chunk_text(self, text, max_tokens=512):
        """Split text into chunks that fit within the maximum token limit."""
        tokens = self.tokenizer.encode(text, truncation=False)

        # Adjust for special tokens
        adjusted_max_tokens = max_tokens - 2

        if len(tokens) <= max_tokens:
            # If the text is already within the token limit, no need to split
            return [text]

        # Split tokens into chunks of size `adjusted_max_tokens`
        chunks = [
            tokens[i: i + adjusted_max_tokens]
            for i in range(0, len(tokens), adjusted_max_tokens)
        ]

        # Decode each chunk back into text
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def get_embedding(self, text):
        """Get the embedding vector for the given text."""
        text_chunks = self.chunk_text(text, max_tokens=512)  # Ensure valid chunks

        embeddings = []
        for chunk in text_chunks:
            response = requests.post(
                self.api_url,
                json={"inputs": [chunk]},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                embedding = response.json()
                if embedding:
                    embeddings.append(embedding[0])
            else:
                raise ValueError(f"Failed to get embedding for chunk: {response.text}")

        # Combine embeddings by averaging across chunks
        combined_embedding = [
            sum(x) / len(x) for x in zip(*embeddings)
        ]
        return combined_embedding

    def ingest_to_db(self, df, table_name, vector_dimension):
        with self.conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                embedding VECTOR({vector_dimension}) -- Requires pgvector extension
            )
            """)
            self.conn.commit()

            # Process each title in the DataFrame
            for title in df['Plot']:
                text_chunks = self.chunk_text(title)  # Split into valid chunks

                embeddings = []
                for chunk in text_chunks:
                    embedding = self.get_embedding(chunk)  # Get embedding for each chunk
                    embeddings.append(embedding)

                # Combine embeddings by averaging across chunks
                combined_embedding = [
                    sum(x) / len(x) for x in zip(*embeddings)
                ]

                cur.execute(
                    f"INSERT INTO {table_name} (title, embedding) VALUES (%s, %s)",
                    (title, combined_embedding)
                )
            self.conn.commit()
        print(f"Data ingested successfully into table {table_name}.")

    def query_most_similar(self, input_text, table_name, top_k=1):
        """Query the most similar embeddings from the database."""
        embedding = self.get_embedding(input_text)  # Get embedding for the input text
        with self.conn.cursor() as cur:
            cur.execute(f"""
            SELECT title, embedding <-> %s::VECTOR AS similarity
            FROM {table_name}
            ORDER BY similarity ASC
            LIMIT %s
            """, (embedding, top_k))
            results = cur.fetchall()
        return results


if __name__ == "__main__":
    # URL of the Hugging Face Text Embedding Inference API (running locally)
    API_URL = "http://localhost:8080/embed"

    # PostgreSQL connection configuration
    DB_CONFIG = {
        "database": "vectordb",
        "user": "postgres",
        "password": "password",
        "host": "127.0.0.1",
        "port": 5432
    }
    TABLE_NAME = "plot_embeddings"
    VECTOR_DIMENSION = 1024  # Match the model's embedding size

    # Read CSV data
    data = pd.read_csv('viblo_data.csv')
    df = pd.DataFrame(data)

    embedder = TextEmbedder(API_URL, DB_CONFIG)
    try:
        embedder.ingest_to_db(df, TABLE_NAME, VECTOR_DIMENSION)

        # Query most similar titles
        input_text = input("Enter a text to find similar entries: ")
        results = embedder.query_most_similar(input_text, TABLE_NAME, top_k=10)

        for result in results:
            print(f"Title: {result[0]}, Similarity: {result[1]}")
    finally:
        embedder.close()
