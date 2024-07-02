# get_embedding_function.py

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings

def get_embedding_function():
    # Choose one of the embedding methods to test
    # Example using OllamaEmbeddings
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings

# Function for testing the embedding function
# def test_embedding_function():
#     embeddings = get_embedding_function()
#     sample_text = "This is a sample text."
#     embedding = embeddings._embed(sample_text)
#     print("Sample Text:", sample_text)
#     print("Embedding:", embedding)
