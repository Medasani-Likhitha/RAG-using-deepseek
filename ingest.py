import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dotenv import load_dotenv

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL")
api_key = os.getenv("QDRANT_API")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_TYPE = ExportType.DOC_CHUNKS
FILE_PATH = "/home/medasanilikhitha/PycharmProjects/RAG-using-deepseek/data/DeepSeek_R1.pdf"

def create_vector_database():
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)
    )

    docling_documents = loader.load()

    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docling_documents
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_one=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [split for doc in docling_documents for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Invalid export type: {EXPORT_TYPE}")

    with open('/home/medasanilikhitha/PycharmProjects/RAG-using-deepseek/data/output_docling.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in docling_documents:
            f.write(doc.page_content + '\n')

    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # Create and persist a Qdrant vector database from the chunked documents
    # vectorstore = QdrantVectorStore.from_documents(
    #     documents=splits,
    #     embedding=embeddings,
    #     url=qdrant_url,
    #     collection_name="rag",
    # )

    # QDRANT CLOUD
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=qdrant_cloud_url,
        api_key=api_key,
        collection_name="rag_cloud",
    )

    print("Vector database created successfully")


if __name__ == "__main__":
    create_vector_database()

