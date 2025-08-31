import streamlit as st
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration and Setup ---
UPLOAD_DIRECTORY = "temp_data_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Define the system prompt for the agent
SYSTEM_PROMPT = """
You are a highly skilled Data Analyst AI Assistant. Your purpose is to help users analyze and understand their CSV data.
You have access to a set of specialized tools to perform your tasks.

Your core workflow is as follows:
1.  **Acknowledge and Greet:** Start by greeting the user and asking them to upload a CSV file if one is not already present.
2.  **File Discovery:** If a user asks about available files, use the `list_uploaded_csv_files` tool to see what's in the `{UPLOAD_DIRECTORY}` directory.
3.  **Initial Analysis:** When a new CSV file is uploaded, first use the `analyze_csv_metadata` tool to get a comprehensive overview of the dataset. This includes column names, data types, missing values, and key statistics. This step is crucial for understanding the data's structure and contents.
4.  **Preview Data:** You can use the `get_csv_preview` tool to quickly display a few sample rows to the user, providing a glimpse of the data's format and content.
5.  **Execution and Computation:** For specific analysis tasks (e.g., "what's the average sales?", "find the top 5 customers"), use the `run_pandas_code` tool. This tool allows you to execute Python code (specifically with pandas) to perform calculations, aggregations, and data manipulations.
    - **Crucial Note:** When using `run_pandas_code`, ensure your code operates on a DataFrame named `df`. Your code should produce a result that can be directly displayed or reasoned upon.
    - **Example:** To find the top 5 product categories by sales, you would use: `run_pandas_code("your_file.csv", "df.groupby('Category')['Sales'].sum().nlargest(5)")`.
6.  **Provide Insights:** After performing an analysis, present your findings clearly and concisely. Explain what the data shows and answer the user's question directly.
7.  **Maintain Context:** Remember the user's uploaded file and the context of the conversation. You don't need to re-analyze metadata for every question unless explicitly asked to or if a new file is uploaded.
8.  **Error Handling:** If a tool call fails, inform the user about the error and suggest a possible solution.
9.  **Output Format:** When providing reports or detailed analysis, you MUST format your response using Markdown. Use headings, bullet points, and bold text for readability.

### Example of Desired Output Format:
---
I have generated a sales report based on the 'Superstore-Data.csv' file. Here are the key insights for strategic business decisions:

#### Sales Report
- **Overall Sales Performance:** Total Sales: $2,297,200.86 | Average Sales per Order: $230.00
- **Monthly Sales Trends:** Sales show a generally upward trend throughout the year, with significant peaks in November and December, suggesting a strong holiday season or year-end push.

#### Top Product Categories
- **Top 5 Product Categories by Sales:**
    - Technology: $837,575.24
    - Furniture: $741,999.95
    - Office Supplies: $717,736.67
- **Top 5 Sub-Categories by Sales:**
    - Phones: $31,036.50
    - Chairs: $30,809.09
    - Storage: $21,809.64

---
Do not attempt to generate or execute arbitrary Python code directly. You must rely exclusively on the provided tools to interact with the CSV data.

Start the conversation by politely asking the user to upload a CSV file.
"""

# --- Streamlit UI Components and State Management ---
st.set_page_config(page_title="Data Analyst Assistant", layout="wide")
st.title("📊 CSV Data Analyst Assistant")
st.info("I am a helpful AI assistant ready to analyze your CSV files. Please upload a file to get started!")

# --- Async Agent Initialization ---
# Use a global variable to ensure the event loop is started only once.
# This prevents the "Event loop is closed" error.
# A custom loop is needed because Streamlit's `asyncio.run` creates and closes a new loop.
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Wrap async functions for Streamlit's sync context
def run_async(coro):
    return st.session_state.loop.run_until_complete(coro)

@st.cache_resource(show_spinner="Starting up the AI data analyst...")
def initialize_agent():
    """Initializes the Multi-Server MCP client and LangChain agent."""
    try:
        mcp_client = MultiServerMCPClient({
            "csv_analyzer": {"command": "python3", "args": ["tools/csv_tools.py"], "transport": "stdio"},
            "local_python_executor": {"command": "python3", "args": ["tools/local_python_executor.py"], "transport": "stdio"}
        })
        tools = run_async(mcp_client.get_tools())
        llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
        agent = create_react_agent(llm, tools)
        return mcp_client, agent
    except Exception as e:
        st.error(f"Failed to initialize the agent. Please check your tool scripts and dependencies. Error: {e}")
        logger.error(f"Agent initialization error: {e}", exc_info=True)
        return None, None

# Initialize session state for agent and chat history
if "agent" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.mcp_client, st.session_state.agent = initialize_agent()

# Sidebar for file upload and controls
with st.sidebar:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.header("Controls")
    clear_chat = st.button("Clear Chat")
    
    if clear_chat:
        st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
        for filename in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
        st.info("All uploaded files and chat history have been cleared. You can now upload a new file.")
        st.rerun()

def display_chat_history():
    """Displays the chat history in the main UI area."""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, SystemMessage):
            # Don't display the system prompt to the user
            continue
        else: # Assumed to be AIMessage
            with st.chat_message("assistant"):
                st.markdown(message.content)

# --- Main App Logic ---
if st.session_state.agent:
    # Handle file upload
    if uploaded_file:
        try:
            # Create a temporary file to save the uploaded content
            temp_file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
            
            # Check if file already exists with the same content
            file_exists = False
            if os.path.exists(temp_file_path):
                # Simple check: compare file sizes
                if os.path.getsize(temp_file_path) == uploaded_file.size:
                    file_exists = True
            
            if not file_exists:
                # Save the new file
                with open(temp_file_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
                
                logger.info(f"File saved to {temp_file_path}")
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.chat_history.append(
                    SystemMessage(content=f"User has uploaded a new CSV file: '{uploaded_file.name}'")
                )
                
                # Automatically send a message to the agent to start analysis
                with st.spinner("Processing your file..."):
                    run_async(
                        st.session_state.agent.ainvoke(
                            {"messages": [HumanMessage(content=f"Analyze the newly uploaded file named '{uploaded_file.name}'")]},
                            config={"configurable": {"thread_id": "1"}}
                        )
                    )
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            logger.error(f"File upload error: {e}", exc_info=True)

    # Display chat history
    display_chat_history()

    # Chat input box
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the agent with the entire chat history
                    response = run_async(
                        st.session_state.agent.ainvoke(
                            {"messages": st.session_state.chat_history},
                            config={"configurable": {"thread_id": "1", "recursion_limit": 50}}
                        )
                    )
                    
                    # The response is an AIMessage with content and tool_calls
                    agent_message = response['messages'][-1]
                    st.markdown(agent_message.content)
                    st.session_state.chat_history.append(agent_message)
                    
                except Exception as e:
                    st.error(f"An error occurred during agent invocation: {e}")
                    logger.error(f"Agent invocation error: {e}", exc_info=True)
else:
    st.warning("Agent initialization failed. Please check the console for details.")