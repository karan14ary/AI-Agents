import requests
from bs4 import BeautifulSoup
import streamlit as st
import os

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# ==============================
# 🔧 CONFIG
# ==============================
FAISS_PATH = "faiss_index"

# Load LLM
llm = OllamaLLM(model="mistral")  # or llama3

# Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# 📦 LOAD / CREATE VECTOR DB
# ==============================
def load_vector_db():
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

vector_db = load_vector_db()

# ==============================
# 🌍 SCRAPER
# ==============================
def scrape_website(url):
    try:
        st.write(f"🌍 Scraping: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.extract()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        return text[:10000]  # limit size

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# ==============================
# 💾 STORE DATA
# ==============================
def store_in_faiss(text, url):
    global vector_db

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    docs = [
        Document(page_content=chunk, metadata={"source": url})
        for chunk in chunks
    ]

    if vector_db is None:
        vector_db = FAISS.from_documents(docs, embeddings)
    else:
        vector_db.add_documents(docs)

    # Save locally
    vector_db.save_local(FAISS_PATH)

    return f"✅ Stored {len(chunks)} chunks."

# ==============================
# 🔍 RETRIEVE + ANSWER
# ==============================
def retrieve_and_answer(query):
    global vector_db

    if vector_db is None:
        return "⚠️ No data available. Add a website first."

    docs = vector_db.similarity_search(query, k=3)

    if not docs:
        return "🤖 No relevant results found."

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content="Answer ONLY from the given context. If unknown, say you don't know."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = llm.invoke(messages)

    return response, docs

# ==============================
# 🎨 STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Web Scraper", layout="wide")

st.title("🤖 AI Web Scraper + FAISS (RAG)")
st.write("Scrape websites → Store → Ask questions")

# Input URL
url = st.text_input("🔗 Enter Website URL")

if st.button("Scrape & Store"):
    if not url:
        st.warning("Enter a URL first")
    else:
        text = scrape_website(url)

        if text:
            msg = store_in_faiss(text, url)
            st.success(msg)
        else:
            st.error("Failed to scrape content.")

# Ask Question
query = st.text_input("❓ Ask a question")

if st.button("Get Answer"):
    if not query:
        st.warning("Enter a question")
    else:
        result = retrieve_and_answer(query)

        if isinstance(result, str):
            st.write(result)
        else:
            answer, docs = result

            st.subheader("🤖 Answer")
            st.write(answer)

            st.subheader("📚 Sources")
            for doc in docs:
                st.write(f"🔗 {doc.metadata['source']}")