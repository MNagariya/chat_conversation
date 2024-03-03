import weaviate
import spacy
import weaviate.classes as wvc
import os

# Set these environment variables
URL = os.getenv("WEAVIATE_URL")
APIKEY = os.getenv("WEAVIATE_API_KEY")

# Connect to a WCS instance
client = weaviate.connect_to_wcs(
    cluster_url=URL,
    auth_credentials=weaviate.auth.AuthApiKey(APIKEY),
    skip_init_checks=True,
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Replace with your inference API key
    },
)


def create_embeddings(text):
    nlp = spacy.load("en_core_web_md")  # Medium model for English
    doc = nlp(text)
    return doc.vector


def store_in_weaviate(embeddings, text):

    client.collections.delete_all()
    client.collections.create(
        name="TextDocument",
        description="Attention Is All You Need",
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
            ef=128,
            max_connections=64,
        ),
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
            vectorize_collection_name=True,
        ),
        properties=[
            wvc.config.Property(
                data_type=wvc.config.DataType.TEXT,
                description="Attention Is All You Need",
                name="title",
                index_filterable=True,
                index_searchable=True,
                skip_vectorization=False,
                vectorize_property_name=False,
            ),
            wvc.config.Property(
                data_type=wvc.config.DataType.TEXT,
                description="The content of the article",
                name="content",
                index_filterable=True,
                index_searchable=True,
                skip_vectorization=False,
                vectorize_property_name=False,
            ),
        ],
        sharding_config=wvc.config.Configure.sharding(
            virtual_per_physical=128,
            desired_count=1,
            desired_virtual_count=128,
        ),
    )
