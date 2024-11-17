import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
import psycopg2
import requests

class TextEmbedder:
    def __init__(self, api_url, db_config, model_name="bert-base-uncased"):
        self.api_url = api_url
        self.conn = psycopg2.connect(**db_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embedding(self, text):
        response = requests.post(
            self.api_url,
            json={"inputs": [text]},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            embeddings = response.json()
            if len(embeddings) > 0:
                return embeddings[0]
        raise ValueError(f"Failed to get embedding: {response.text}")

    def query_most_similar(self, input_text, table_name, top_k=5):
        embedding = self.get_embedding(input_text)
        with self.conn.cursor() as cur:
            cur.execute(f"""
            SELECT title, 1 - (embedding <-> %s::VECTOR) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT %s
            """, (embedding, top_k))
            results = cur.fetchall()
        return results

# Streamlit app
def main():
    st.title("Text Similarity Finder")

    # Sidebar inputs for database configuration and API URL
    st.sidebar.header("Database Configuration")
    db_name = st.sidebar.text_input("Database Name", "vectordb")
    db_user = st.sidebar.text_input("User", "postgres")
    db_password = st.sidebar.text_input("Password", type="password")
    db_host = st.sidebar.text_input("Host", "127.0.0.1")
    db_port = st.sidebar.text_input("Port", "5432")
    table_name = st.sidebar.text_input("Table Name", "text_embeddings")
    api_url = st.sidebar.text_input("API URL", "http://localhost:8080/embed")

    if st.sidebar.button("Connect to Database"):
        db_config = {
            "database": db_name,
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port
        }
        embedder = TextEmbedder(api_url, db_config)
        st.session_state["embedder"] = embedder
        st.success("Connected to the database!")

    # Text input for similarity search
    input_text = st.text_area("Enter a text to search for similar entries:")
    top_k = st.slider("Number of results to return", 1, 20, 5)

    if st.button("Find Similar Texts"):
        if "embedder" in st.session_state:
            embedder = st.session_state["embedder"]
            try:
                results = embedder.query_most_similar(input_text, table_name, top_k=top_k)
                st.subheader("Similar Texts:")
                for idx, (title, similarity) in enumerate(results, 1):
                    st.write(f"{idx}. **{title}** - Similarity: {similarity:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please connect to the database first!")

if __name__ == "__main__":
    main()
