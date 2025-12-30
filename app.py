{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader\
from langchain_text_splitters import RecursiveCharacterTextSplitter\
from langchain_community.vectorstores import FAISS\
from langchain_openai import OpenAIEmbeddings, ChatOpenAI\
from langchain.chains import create_retrieval_chain\
from langchain.chains.combine_documents import create_stuff_documents_chain\
from langchain_core.prompts import ChatPromptTemplate\
\
# --- CONFIGURATION & BRANDING ---\
st.set_page_config(page_title="Geostrata AI", page_icon="\uc0\u55356 \u57103 ", layout="wide")\
\
# Custom Styling for The Geostrata (Black/Gold theme)\
st.markdown("""\
<style>\
    .stApp \{ background-color: #0e1117; color: #ffffff; \}\
    h1 \{ color: #d4af37; \}\
    .stButton>button \{ background-color: #d4af37; color: black; border-radius: 5px; font-weight: bold;\}\
    a \{ color: #d4af37; \}\
</style>\
""", unsafe_allow_html=True)\
\
st.title("\uc0\u55356 \u57103  The Geostrata AI Analyst")\
st.markdown("Answers questions on geopolitics using **only** The Geostrata's Articles and YouTube videos.")\
\
# --- SIDEBAR: SETTINGS & DATA ---\
with st.sidebar:\
    st.header("\uc0\u9881 \u65039  Configuration")\
    api_key = st.text_input("OpenAI API Key", type="password", help="Get this from platform.openai.com")\
    \
    st.divider()\
    st.subheader("\uc0\u55357 \u56538  Knowledge Base")\
    st.info("Paste URLs below to train the bot.")\
    \
    # Pre-filled with your actual main pages\
    default_web = "https://www.thegeostrata.com/geopost\\nhttps://www.thegeostrata.com/aboutus"\
    website_urls = st.text_area("Article URLs (One per line)", value=default_web, height=150)\
    \
    # Pre-filled with your actual channel videos\
    default_yt = "https://www.youtube.com/watch?v=hBpdp28CdIU\\nhttps://www.youtube.com/watch?v=nYpkLo_m--E"\
    youtube_urls = st.text_area("YouTube Video URLs (One per line)", value=default_yt, height=150)\
    \
    train_btn = st.button("\uc0\u55357 \u56580  Update Knowledge Base")\
\
# --- AI LOGIC ---\
def get_vectorstore(web_urls, yt_urls):\
    """Scrapes data and prepares the memory."""\
    docs = []\
    \
    # 1. Scrape Websites\
    web_list = [u.strip() for u in web_urls.split('\\n') if u.strip()]\
    if web_list:\
        try:\
            # User agent helps bypass some website blockers\
            loader = WebBaseLoader(web_list, header_template=\{'User-Agent': 'Mozilla/5.0'\})\
            docs.extend(loader.load())\
        except Exception as e:\
            st.error(f"Error loading website: \{e\}")\
\
    # 2. Scrape YouTube Transcripts\
    yt_list = [u.strip() for u in yt_urls.split('\\n') if u.strip()]\
    if yt_list:\
        try:\
            for video in yt_list:\
                loader = YoutubeLoader.from_youtube_url(video, add_video_info=True)\
                docs.extend(loader.load())\
        except Exception as e:\
            st.error(f"Error loading YouTube: \{e\}")\
            \
    if not docs:\
        return None\
\
    # 3. Split text into chunks\
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
    split_docs = text_splitter.split_documents(docs)\
    \
    # 4. Save to Vector Store (Memory)\
    return FAISS.from_documents(split_docs, OpenAIEmbeddings(api_key=api_key))\
\
def get_rag_chain(vectorstore):\
    """Creates the Indian-Perspective AI Brain."""\
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)\
\
    system_prompt = (\
        "You are a Senior Strategic Analyst for 'The Geostrata', a youth-led Indian think tank. "\
        "Your goal is to answer questions using strictly the provided context.\\n\\n"\
        "**INSTRUCTIONS:**\\n"\
        "1. **Perspective:** ALWAYS frame your answer from an **Indian Strategic Perspective**. "\
        "Highlight concepts like Strategic Autonomy, Multi-alignment, Global South leadership, and Neighborhood First.\\n"\
        "2. **Tone:** Professional, objective, yet assertive.\\n"\
        "3. **Citations:** You MUST cite sources. If you use a video, mention its title. If an article, mention the link.\\n"\
        "4. **Unknowns:** If the provided context doesn't contain the answer, say 'The Geostrata has not covered this topic yet.'\\n\\n"\
        "Context: \{context\}"\
    )\
    \
    prompt = ChatPromptTemplate.from_messages([\
        ("system", system_prompt),\
        ("human", "\{input\}"),\
    ])\
    \
    retriever = vectorstore.as_retriever()\
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))\
    return chain\
\
# --- MAIN APP ---\
if train_btn and api_key:\
    with st.spinner("Scouring The Geostrata archives... (This may take a moment)"):\
        st.session_state.vectorstore = get_vectorstore(website_urls, youtube_urls)\
        st.success("\uc0\u9989  Training Complete! The AI now knows your content.")\
\
if "vectorstore" in st.session_state and st.session_state.vectorstore:\
    user_query = st.chat_input("Ask a question (e.g., 'What is India's stance on the Chahbahar port?')")\
    \
    if user_query:\
        # Display User Message\
        st.chat_message("user").write(user_query)\
        \
        # Generate Answer\
        with st.chat_message("assistant"):\
            with st.spinner("Analyzing from Indian perspective..."):\
                chain = get_rag_chain(st.session_state.vectorstore)\
                response = chain.invoke(\{"input": user_query\})\
                \
                st.write(response["answer"])\
                \
                # Citation Dropdown\
                with st.expander("\uc0\u55357 \u56538  Sources Used"):\
                    for doc in response["context"]:\
                        source = doc.metadata.get('source', 'Unknown')\
                        title = doc.metadata.get('title', 'Untitled')\
                        st.write(f"- [\{title\}](\{source\})")\
elif not api_key:\
    st.warning("\uc0\u55357 \u56392  Please enter your OpenAI API Key in the sidebar to start.")}