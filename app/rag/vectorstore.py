
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# Hugging Face embedding model dimension
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    if normalized not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unsupported embedding model: {name}"
        )

    return EMBEDDING_DIMENSIONS[normalized]


# Initialize local Hugging Face embeddings
def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print("Loading Hugging Face embedding model...")

        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Hugging Face model loaded successfully!")

    return _embeddings


# Create or validate Pinecone index
def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)

    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:

        print("Pinecone index already exists.")

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info, "dimension", None
        )

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension != desired_dimension:
            raise ValueError(
                f"Pinecone dimension mismatch! "
                f"Existing: {current_dimension}, "
                f"Expected: {desired_dimension}. "
                "Please use a new index name."
            )

    else:

        print("Creating Pinecone index...")
        print("Index name:", settings.pinecone_index_name)
        print("Dimension:", desired_dimension)

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

        print("Pinecone index created successfully!")

    return pc.Index(settings.pinecone_index_name)


# Initialize Pinecone vector store
def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# Retrieve relevant documents
def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


# Add documents to Pinecone
def add_documents(chunks):

    if not chunks:
        raise ValueError("No document chunks provided for ingestion.")

    store = get_vectorstore()

    print(f"Adding {len(chunks)} chunks to Pinecone...")

    ids = store.add_documents(chunks)

    print(f"Successfully added {len(ids)} chunks!")

    return ids
