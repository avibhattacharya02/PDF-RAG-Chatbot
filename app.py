import html
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from google import genai

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Papertrail — PDF reader", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

    :root {
        --paper: #f7f6f2;
        --surface: #fffefa;
        --ink: #252822;
        --muted: #777a70;
        --line: #e6e4dc;
        --green: #46634f;
        --green-soft: #e9eee8;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1100px; padding: 3.2rem 3rem 7rem; }
    section[data-testid="stSidebar"] {
        background: #f0efe9;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] > div { padding-top: 2rem; }
    .brand-mark {
        color: var(--green); font-size: .72rem; font-weight: 700;
        letter-spacing: .17em; text-transform: uppercase; margin-bottom: 2.4rem;
    }
    .sidebar-title {
        color: var(--ink); font: 600 1.55rem 'Newsreader', Georgia, serif;
        margin: 0 0 .3rem;
    }
    .sidebar-copy { color: var(--muted); font-size: .87rem; line-height: 1.55; margin: 0 0 1.4rem; }
    .section-label {
        color: var(--muted); font-size: .68rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; margin: 1.5rem 0 .55rem;
    }
    .page-kicker {
        color: var(--green); font-size: .7rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .8rem;
    }
    .page-title {
        color: var(--ink); font: 500 clamp(2.8rem, 5vw, 4.4rem)/1.02 'Newsreader', Georgia, serif;
        letter-spacing: -.045em; margin: 0; max-width: 760px;
    }
    .page-subtitle { color: var(--muted); font-size: 1rem; margin: 1rem 0 1.8rem; }
    .page-rule { border-top: 1px solid var(--line); margin: 0 0 2rem; }
    .ready-chip {
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .42rem .7rem; border: 1px solid var(--line); border-radius: 99px;
        background: var(--surface); color: var(--muted); font-size: .75rem;
    }
    .ready-dot { width: 7px; height: 7px; background: #b9b9ad; border-radius: 50%; }
    .ready-dot.on { background: #63876b; }
    .empty-state {
        border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
        padding: 2.2rem 2.4rem; margin: 1.2rem 0 1rem;
    }
    .empty-eyebrow { color: var(--green); font-size: .68rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    .empty-title { color: var(--ink); font: 500 1.75rem 'Newsreader', Georgia, serif; margin: .55rem 0 .4rem; }
    .empty-copy { color: var(--muted); font-size: .9rem; line-height: 1.65; max-width: 560px; margin: 0; }
    .stButton > button {
        border-radius: 7px; border: 1px solid var(--green); background: var(--green);
        color: white; font-weight: 600; min-height: 2.65rem;
    }
    .stButton > button:hover { border-color: #354c3c; background: #354c3c; color: white; }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #c9c9bd; border-radius: 8px; background: rgba(255,254,250,.65);
    }
    [data-testid="stFileUploader"] { color: var(--ink); }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small { color: var(--muted) !important; }
    [data-testid="stFileUploader"] button {
        min-height: 2.6rem; padding: .45rem 1rem; border-radius: 7px !important;
        border: 1px solid var(--green) !important; background: var(--green) !important;
        color: #fff !important; font-weight: 600;
    }
    [data-testid="stFileUploader"] button * { color: #fff !important; fill: #fff !important; }
    [data-testid="stFileUploader"] button:hover { background: #354c3c !important; }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
        color: var(--ink) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
        color: var(--ink) !important;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: var(--paper) !important; background-image: none !important;
    }
    [data-testid="stBottomBlockContainer"] { padding-top: .6rem !important; }
    [data-testid="stChatInput"] {
        max-width: 820px; margin: 0 auto; border: 1px solid #d9d8cf !important;
        border-radius: 10px !important; background: var(--surface) !important;
        box-shadow: 0 2px 10px rgba(37, 40, 34, .04) !important;
    }
    [data-testid="stChatInput"] > div { background: var(--surface) !important; border-radius: 10px !important; }
    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
        background: transparent !important; caret-color: var(--green);
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1;
    }
    [data-testid="stChatInput"] button {
        background: var(--green) !important; color: #fff !important;
        border: 0 !important; border-radius: 7px !important;
    }
    [data-testid="stChatInput"] button:hover { background: #354c3c !important; }
    [data-testid="stChatInput"] button svg { fill: #fff !important; color: #fff !important; }
    @media (max-width: 700px) {
        .block-container { padding: 2rem 1.1rem 6rem; }
        .page-title { font-size: 2.8rem; }
        .empty-state { padding: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for vector store and chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_name" not in st.session_state:
    st.session_state.document_name = None


@st.cache_resource
def get_gemini_client():
    """Reuse one Gemini client across Streamlit reruns and chat requests."""
    return genai.Client()


class GeminiEmbeddings(Embeddings):
    """LangChain embeddings adapter backed by the Google Gen AI SDK."""

    def __init__(self):
        self.model = "gemini-embedding-001"

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), 100):
            response = get_gemini_client().models.embed_content(
                model=self.model,
                contents=texts[start:start + 100],
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            vectors.extend(embedding.values for embedding in response.embeddings)
        return vectors

    def embed_query(self, text):
        response = get_gemini_client().models.embed_content(
            model=self.model,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return response.embeddings[0].values


def process_pdf(uploaded_file):
    """Processes the PDF, splits text into chunks, and creates a Chroma vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # 1. Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 2. Chunk Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Create Gemini embeddings & vector store
        embeddings = GeminiEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        return vector_store

    finally:
        os.remove(tmp_path)


# Document controls
with st.sidebar:
    st.markdown('<div class="brand-mark">Papertrail · Reader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Your document</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-copy">Add a PDF to begin. Your questions will be answered from its pages.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Choose a file</div>', unsafe_allow_html=True)
    pdf_doc = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_doc:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF and generating embeddings..."):
                st.session_state.vector_store = process_pdf(pdf_doc)
                st.session_state.document_name = pdf_doc.name
                st.success("PDF processed successfully!")

    if st.session_state.document_name:
        safe_document_name = html.escape(st.session_state.document_name)
        st.markdown(
            f'<div class="ready-chip"><span class="ready-dot on"></span>{safe_document_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ready-chip"><span class="ready-dot"></span>No document loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">A quick note</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-copy">For the best results, ask specific questions about names, dates, or ideas in the document.</p>',
        unsafe_allow_html=True,
    )

# Page introduction
st.markdown(
    '<div class="page-kicker">A quieter way to work through PDFs</div>'
    '<h1 class="page-title">Read the details.<br>Find what matters.</h1>'
    '<p class="page-subtitle">Ask questions about a document and get answers grounded in its pages.</p>'
    '<div class="page-rule"></div>',
    unsafe_allow_html=True,
)

status_text = html.escape(st.session_state.document_name or "Waiting for a document")
status_class = "on" if st.session_state.vector_store is not None else ""
st.markdown(
    f'<div class="ready-chip"><span class="ready-dot {status_class}"></span>{status_text}</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-eyebrow">Start with a document</div>'
        '<div class="empty-title">Make the long PDF a little easier.</div>'
        '<p class="empty-copy">Upload a file in the left panel and process it. Then ask about a section, a person, a date, or anything you want to find again.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query Input
user_query = st.chat_input("Ask a question about your PDF...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if not st.session_state.vector_store:
        response = "Upload and process a PDF in the left panel before I can answer questions about it."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Build RAG Chain
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    relevant_docs = retriever.invoke(user_query)
                    context = "\n\n".join(
                        document.page_content for document in relevant_docs
                    )
                    prompt = (
                        "You answer questions using only the provided PDF context. "
                        "If the answer is not in the context, say you don't know. "
                        "Keep the answer concise and accurate.\n\n"
                        f"PDF context:\n{context}\n\n"
                        f"Question: {user_query}"
                    )
                    gemini_response = get_gemini_client().models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                    )
                    response = gemini_response.text or "Gemini returned an empty response."
                    st.markdown(response)
                except Exception as error:
                    response = f"Gemini request failed: {error}"
                    st.error(response)

        # Append response to history
        st.session_state.messages.append({"role": "assistant", "content": response})