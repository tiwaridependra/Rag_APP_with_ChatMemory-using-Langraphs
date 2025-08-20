
from typing import List, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# ---- State ----
# ---- Setup ----
model = ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=api_key)
# result=model.invoke("what is the capital of india")
# print(result.content)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=api_key)
class State(TypedDict):
    query: str
    context: List[str]
    answer: str
    history: List[str]   # keeps memory of past chats


vector_store = None   

# ---- PDF Ingestion ----
def build_faiss_from_pdf(pdf_path: str):
    """Read PDF, chunk it, and store in FAISS index."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Chunk with Recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Build FAISS
    global vector_store
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# ---- Nodes ----
def retrieve(state: State):
    """Retrieve relevant docs for query."""
    docs = vector_store.similarity_search(state["query"], k=3)
    return {"context": [doc.page_content for doc in docs]}

def generate(state: State):
    """Generate answer using query, context, and history."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use context and chat history."),
        ("human", "History:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}")
    ])
    chain = prompt | model
    answer = chain.invoke({
        "query": state["query"],
        "context": "\n".join(state["context"]),
        "history": "\n".join(state.get("history", []))
    })
    # Add to history
    new_history = state.get("history", []) + [
        f"User: {state['query']}",
        f"AI: {answer.content}"
    ]
    return {"answer": answer.content, "history": new_history}


graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START,"retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
checkpointer=InMemorySaver()
app = graph.compile(checkpointer=checkpointer)



