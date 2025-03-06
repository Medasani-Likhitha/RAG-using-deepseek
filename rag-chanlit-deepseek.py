import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make the answer.
#
# Context: {context}
# Question: {question}
#
# Only return the helpful answer below and nothing else.
# Helpfull answwer:
# """

llm = OllamaLLM(
    model="deepseek-r1:1.5b"
)


@cl.on_chat_start
async def on_chat_start(chat: Chat):
    template = """Answer the question based only on the following context:
    
    {context}
    
    Questiom: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    embedding = HuggingFaceEmbeddings(model_nme=EMBED_MODEL_ID)

    vectorstore  = QdrantVectorStore.from_existing_collection(embedding=embedding, collection_name="rag", url=qdrant_url)

    retriever = vectorstore.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)





























