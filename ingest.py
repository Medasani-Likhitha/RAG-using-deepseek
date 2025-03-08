import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_docling.loader import DoclingLoader
from docling.chunking import HybridChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_docling.loader import ExportType
from dotenv import load_dotenv
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_TYPE = ExportType.DOC_CHUNKS
FILE_PATH = "data/2021-09-01-2021-09-30.json"

def create_vector_database():
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)
    )

    dockling_documents = loader.load()

    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = dockling_documents
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_one=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [split for doc in dockling_documents for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Invalid export type: {EXPORT_TYPE}")

    with open("data/2021-09-01-2021-09-30.json", "r") as f:
        for doc in dockling_documents:
            f.write(doc.page_content + '\n')

    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # Create and persist a Qdrant vector database from the chunked documents
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="rag",
    )

    print("Vector database created successfully")


if __name__ == "__main__":
    create_vector_database()

