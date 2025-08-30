import streamlit as st
import asyncio
import os
import shutil
import uuid
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage

# Load env
load_dotenv()

# Config
CONFIG = {"configurable": {"thread_id": "1"}}

# --- Global event loop (reuse it) ---
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# --- Directory for temporary data files ---
TEMP_DATA_DIR = "temp_data_files"
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# --- Agent Setup ---
@st.cache_resource(show_spinner="Loading agent...")
def get_agent():
    client = MultiServerMCPClient({
        "csv_analyzer": {"command": "python3", "args": ["tools/csv_tools.py"], "transport": "stdio"},
        "local_python_executor": {"command": "python3", "args": ["tools/local_python_executor.py"], "transport": "stdio"}
    })
    tools = loop.run_until_complete(client.get_tools())
    llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
    return create_react_agent(llm, tools)

agent = get_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="CSV Chat Agent", layout="wide")
st.title("📊 CSV Chat Agent")
st.caption("Upload CSV/TXT/JSON files and ask questions about them.")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 File Upload")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["csv", "txt", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(TEMP_DATA_DIR, uploaded_file.name)
            if uploaded_file.name not in st.session_state.uploaded_files:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files[uploaded_file.name] = file_path

    # Button to clear everything
    if st.button("Clear Chat & Files"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.thread_id = str(uuid.uuid4())
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# --- Show uploaded files ---
if st.session_state.uploaded_files:
    st.subheader("📑 Uploaded Files")
    for file_name, file_path in st.session_state.uploaded_files.items():
        st.markdown(f"- **{file_name}** (`{file_path}`)")

# --- Display chat history ---
for msg in st.session_state.chat_history:
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Input box for chatting ---
if user_input := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use a dedicated function for async execution
                async def run_agent():
                    return await agent.ainvoke(
                        {"messages": st.session_state.chat_history},
                        config=CONFIG
                    )
                
                # Run the async function
                response = asyncio.run(run_agent())
                agent_response = response["messages"][-1]
                st.session_state.chat_history.append(agent_response)
                st.markdown(agent_response.content)
                
            except RuntimeError as e:
                if "event loop" in str(e):
                    # Fallback: use a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(agent.ainvoke(
                            {"messages": st.session_state.chat_history},
                            config=CONFIG
                        ))
                        agent_response = response["messages"][-1]
                        st.session_state.chat_history.append(agent_response)
                        st.markdown(agent_response.content)
                    finally:
                        loop.close()
                else:
                    st.error(f"Error: {e}")

