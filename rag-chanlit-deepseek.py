import os
import warnings
import logging
from time import sleep
from httpx import ConnectError, HTTPStatusError

warnings.filterwarnings("ignore")

from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore

import chainlit as cl
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL")
api_key = os.getenv("QDRANT_API")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Ollama Model with Error Handling
try:
    llm = OllamaLLM(model="deepseek-r1:1.5b")  # Make sure this URL is correct
    logger.info("Ollama model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Ollama model: {e}")
    raise e  # Re-raise the exception to halt execution if the model fails to initialize


# Retry decorator for connection failures
def retry_on_failure(func, retries=3, delay=2):
    def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except (ConnectError, HTTPStatusError) as e:
                logger.error(f"Connection failed: {e}. Retrying in {delay} seconds...")
                attempts += 1
                sleep(delay)
        raise Exception(f"Failed to connect after {retries} attempts")
    return wrapper


@cl.on_chat_start
async def on_chat_start():
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # Wrap the vectorstore initialization with retry logic
    @retry_on_failure
    def initialize_vectorstore():
        return QdrantVectorStore.from_existing_collection(
            embedding=embedding, collection_name="qdrant_rag", url=qdrant_cloud_url, api_key=api_key
        )

    vectorstore = initialize_vectorstore()

    retriever = vectorstore.as_retriever()

    runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm  # Ollama model here
            | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

    # # Test Ollama LLM directly to ensure it's working
    # try:
    #     test_response = llm("Test query to check Ollama connectivity")
    #     logger.info(f"Ollama test response: {test_response}")
    # except Exception as e:
    #     logger.error(f"Error testing Ollama model: {e}")

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source = d.metadata.get('source', 'Unknown')  # Safely get 'source' metadata
                page = d.metadata.get('page', 'Unknown')  # Safely get 'page' metadata
                source_page_pair = (source, page)
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            # logger.info(f"Ollama LLM Response: {response}")  # Log the LLM response
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    try:
        async for chunk in runnable.astream(
                message.content,
                config=RunnableConfig(callbacks=[
                    cl.LangchainCallbackHandler(),
                    PostMessageHandler(msg)
                ]),
        ):
            # logger.info(f"LLM Chunk Received: {chunk}")  # Log each chunk from the LLM
            await msg.stream_token(chunk)

    except ConnectError as e:
        logger.error(f"Connection error during message processing: {e}")
        await cl.Message(content="Connection error occurred. Please try again later.").send()

    await msg.send()
