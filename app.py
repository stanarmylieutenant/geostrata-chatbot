import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION & BRANDING ---
st.set_page_config(page_title="Geostrata AI", page_icon="🌏", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1 { color: #d4af37; }
    .stButton>button { background-color: #d4af37; color: black; border-radius: 5px; font-weight: bold;}
    a { color: #d4af37; }
</style>
""", unsafe_allow_html=True)

st.title("🌏 The Geostrata AI Analyst")
st.markdown("Answers questions on geopolitics using **only** The Geostrata's Articles and YouTube videos.")

# --- SIDEBAR: SETTINGS & DATA ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.subheader("📚 Knowledge Base")
    
    default_web = "https://www.thegeostrata.com/geopost\nhttps://www.thegeostrata.com/aboutus"
    website_urls = st.text_area("Article URLs (One per line)", value=default_web, height=150)
    
    default_yt = "https://www.youtube.com/watch?v=hBpdp28CdIU\nhttps://www.youtube.com/watch?v=nYpkLo_m--E"
    youtube_urls = st.text_area("YouTube Video URLs (One per line)", value=default_yt, height=150)
    
    train_btn = st.button("🔄 Update Knowledge Base")

# --- AI LOGIC ---
def get_vectorstore(web_urls, yt_urls):
    """Scrapes data and prepares the memory."""
    docs = []
    
    # 1. Scrape Websites
    web_list = [u.strip() for u in web_urls.split('\n') if u.strip()]
    if web_list:
        try:
            loader = WebBaseLoader(web_list, header_template={'User-Agent': 'Mozilla/5.0'})
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading website: {e}")

    # 2. Scrape YouTube Transcripts
    yt_list = [u.strip() for u in yt_urls.split('\n') if u.strip()]
    if yt_list:
        try:
            for video in yt_list:
                loader = YoutubeLoader.from_youtube_url(video, add_video_info=True)
                docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading YouTube: {e}")
            
    if not docs:
        return None

    # 3. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    # 4. Save to Memory (FAISS)
    return FAISS.from_documents(split_docs, OpenAIEmbeddings(api_key=api_key))

def get_rag_chain(vectorstore):
    """Creates the Indian-Perspective AI Brain."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)

    system_prompt = (
        "You are a Senior Strategic Analyst for 'The Geostrata', a youth-led Indian think tank. "
        "Your goal is to answer questions using strictly the provided context.\n\n"
        "**INSTRUCTIONS:**\n"
        "1. **Perspective:** ALWAYS frame your answer from an **Indian Strategic Perspective**. "
        "Highlight concepts like Strategic Autonomy, Multi-alignment, Global South leadership, and Neighborhood First.\n"
        "2. **Tone:** Professional, objective, yet assertive.\n"
        "3. **Citations:** You MUST cite sources. If you use a video, mention its title. If an article, mention the link.\n"
        "4. **Unknowns:** If the provided context doesn't contain the answer, say 'The Geostrata has not covered this topic yet.'\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    retriever = vectorstore.as_retriever()
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return chain

# --- MAIN APP ---
if train_btn and api_key:
    with st.spinner("Scouring The Geostrata archives..."):
        st.session_state.vectorstore = get_vectorstore(website_urls, youtube_urls)
        st.success("✅ Training Complete! The AI now knows your content.")

if "vectorstore" in st.session_state and st.session_state.vectorstore:
    user_query = st.chat_input("Ask a question...")
    
    if user_query:
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                chain = get_rag_chain(st.session_state.vectorstore)
                response = chain.invoke({"input": user_query})
                st.write(response["answer"])
                
                with st.expander("📚 Sources Used"):
                    for doc in response["context"]:
                        source = doc.metadata.get('source', 'Unknown')
                        title = doc.metadata.get('title', 'Untitled')
                        st.write(f"- [{title}]({source})")

elif not api_key:
    st.warning("👈 Please enter your OpenAI API Key in the sidebar to start.")
