from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from embeddings import client
from pdf_text_extractor import extract_text_from_pdf
from embeddings import create_embeddings
from embeddings import store_in_weaviate
from getpass import getpass
import os
from dotenv import load_dotenv

app = Flask(__name__)
from getpass import getpass

OPENAI_API_KEY = getpass()

openai_model = OpenAI()
load_dotenv()
prompt_template = """
User: {user_input}
Context: {retrieved_context}

Ask the OpenAI model to: {instruction}
"""


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    retrieved_context = retrieve_context_from_weaviate(user_input)

    # Construct prompt using the template
    prompt = prompt_template.format(
        user_input=user_input,
        retrieved_context=retrieved_context,
        instruction="Summarize the retrieved information",
    )

    # Execute the chain and get response
    chain = LLMChain(openai_model, prompt=prompt)
    response = chain.run()

    return response


def retrieve_context_from_weaviate(user_input):
    class_name = "TextDocument"
    return (
        client.collections.get(class_name)
        .query.near_text(user_input, limit=2)
        .objects[0]
        .propertires
    )


if __name__ == "__main__":
    text = extract_text_from_pdf("static/testing.pdf")
    embeddings = create_embeddings(text)
    store_in_weaviate(embeddings, text)
    app.run(debug=True)
