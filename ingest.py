import requests
import pandas as pd
import psycopg2
from transformers import AutoTokenizer


class TextEmbedder:
    def __init__(self, api_url, db_config, model_name="bert-base-uncased"):
        self.api_url = api_url
        self.conn = psycopg2.connect(**db_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize tokenizer

    def get_embedding(self, text):
        response = requests.post(
            self.api_url,
            json={"inputs": [text]},  
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            embeddings = response.json()  # API typically returns an array of embeddings
            if len(embeddings) > 0:
                return embeddings[0]  # Extract the embedding vector for the input text
        raise ValueError(f"Failed to get embedding: {response.text}")

    def chunk_text(self, text, max_tokens=512):
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        current_chunk = []

        for token in tokens:
            if len(current_chunk) + 1 > max_tokens:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = [token]  
            else:
                current_chunk.append(token)

        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
        
        return chunks

    def ingest_to_db(self, df, table_name, vector_dimension):
        with self.conn.cursor() as cur:
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                embedding VECTOR({vector_dimension})  -- Dimension as per your embedding model
            )
            """)
            self.conn.commit()

            for title in df['Title']:
                text_chunks = self.chunk_text(title)
                
                embeddings = []
                for chunk in text_chunks:
                    embedding = self.get_embedding(chunk)
                    embeddings.append(embedding)
                
                combined_embedding = [sum(x) / len(x) for x in zip(*embeddings)] 
                
                cur.execute(
                    f"INSERT INTO {table_name} (title, embedding) VALUES (%s, %s)",
                    (title, combined_embedding)
                )
            self.conn.commit()
        print(f"Data ingested successfully into table {table_name}.")
        
    def query_most_similar(self, input_text, table_name, top_k=1):
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
        "host":"127.0.0.1",
        "port": 5432
    }
    TABLE_NAME = "text_embeddings"
    VECTOR_DIMENSION = 1024  


    data = pd.read_csv('viblo_title.csv')
    df = pd.DataFrame(data)


    embedder = TextEmbedder(API_URL, DB_CONFIG)
    # embedder.ingest_to_db(df, TABLE_NAME, VECTOR_DIMENSION)

    input_text = input()
    results = embedder.query_most_similar(input_text, TABLE_NAME, top_k=10)

    for result in results:
        print(f"Title: {result[0]}, Similarity: {result[1]}")