import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context if possible, else answer it from your general knowledge, give me what your answer if you think it could help more than the context I gave you, and mention that it's not from the given context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Generated Prompt:")
    print(prompt)  # Check the generated prompt for correctness

    # Load the Hugging Face model and tokenizer
    model_name = "adamfendri/fine-tuned-distilbert-medical-chatbot"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Create a pipeline for question answering
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Use the pipeline to get the answer
    qa_input = {
        "question": query_text,
        "context": context_text
    }
    response = qa_pipeline(qa_input)

    # Format and print the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response['answer']}\nSources: {sources}"
    print(formatted_response)
    return response['answer']

if __name__ == "__main__":
    main()