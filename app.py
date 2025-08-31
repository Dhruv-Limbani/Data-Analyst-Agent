import streamlit as st
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import shutil
import logging
import base64
from pathlib import Path
import re

# Get the absolute path to the directory where the tools are located
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")

# --- Configuration and Setup ---
UPLOAD_DIRECTORY = os.path.join(TOOLS_DIR, "temp_data_files")
PLOT_DIRECTORY = os.path.join(TOOLS_DIR, "temp_plot_images")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- HACK: Base64 and HTML for images ---
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    # Adjusting for different image formats if needed
    img_type = "image/png"
    if img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
        img_type = "image/jpeg"
    img_html = f"<p align='center'><img src='data:{img_type};base64,{img_to_bytes(img_path)}' style='max-width:100%; height:auto;'></p>"
    return img_html
    
# --- display_message_content for interleaved output ---
import re

def display_message_content(message_content):
    if isinstance(message_content, str):
        # Use regex to find all potential image paths within the string
        # This will work regardless of where the path appears
        message_content = message_content.replace("/Users/dhruv/Projects/Data Analyst Agent/","")
        image_paths = re.findall(r"(\S+temp_plot_images/\S+\.png)", message_content)
        
        # Replace each found path with the Base64-encoded HTML
        for path in image_paths:
            if os.path.exists(path):
                image_html = img_to_html(path)
                message_content = message_content.replace(path, image_html)
        
        st.markdown(message_content, unsafe_allow_html=True)
        
    elif isinstance(message_content, list):
        full_text_with_images = ""
        for item in message_content:
            if isinstance(item, str):
                image_paths = re.findall(r"(\S+temp_plot_images/\S+\.png)", item)
                if image_paths and os.path.exists(image_paths[0]):
                    # If an image path is found, replace it with HTML
                    html_content = img_to_html(image_paths[0])
                    # Re-render the string after replacing the path
                    full_text_with_images += item.replace(image_paths[0], html_content)
                else:
                    full_text_with_images += item
            else:
                full_text_with_images += str(item)
            
            full_text_with_images += "\n" # Add a newline to separate list items
        
        st.markdown(full_text_with_images, unsafe_allow_html=True)
        
    else:
        st.markdown(str(message_content))

# Define the system prompt for the agent
SYSTEM_PROMPT = f"""
You are a highly skilled Data Analyst AI Assistant. Your purpose is to help users analyze and understand their CSV data. You have access to a set of specialized tools to perform your tasks.

Your core workflow is as follows:
1.  **Acknowledge and Greet:** Start by greeting the user and asking them to upload a CSV file if one is not already present in {UPLOAD_DIRECTORY}. You check that by using the `list_uploaded_csv_files` tool.
2. If a file is present, use the `analyze_csv_metadata` tool to get an overview of the dataset.
3.  **File Discovery:** If a user asks about available files, use the `list_uploaded_csv_files` tool to see what's in the `{UPLOAD_DIRECTORY}` directory.
4.  **Initial Analysis:** When a new CSV file is uploaded, first use the `analyze_csv_metadata` tool to get a comprehensive overview of the dataset. This step is crucial for understanding the data's structure and contents.
5.  **Execution and Computation:** For specific analysis tasks, use the `run_pandas_code` tool. Your code must operate on the preloaded DataFrame named `df` and the tool's output will be your result.
6.  **Data Visualization:** Consider creating visualizations (e.g., charts, graphs) using the `run_plotting_code` tool whenever a plot would significantly enhance the understanding of the data or a particular insight, in addition to explicit user requests. This tool will create and save one or more plots to {PLOT_DIRECTORY}, returning their file paths. Your final response should include these images.
Note: All the necessary libraries (`numpy` as `np`, `pandas` as `pd, `matplotlib.pyplot` as `plt` and `seaborn` as `sns`) are already imported and available in the execution environment. **You MUST NOT attempt to import them or use any form of the `__import__` function within your code.** Your code should directly use `np`,`pd`,`plt` or `sns` to create visualizations or perform any analysis.
7.  **Provide a Final Response:** After gathering all necessary data using your tools, stop making tool calls. Synthesize a single, clear, and comprehensive response. The response can contain both text and image references (image paths) to provide a complete answer. Do not output raw tool call results.

### Output Formatting Instructions
You MUST format your final response using Markdown. Follow this strict structure:

-   Begin with a clear introductory sentence.
-   Use Markdown headings (e.g., `##`, `###`) to structure your report.
-   Use bold text (`**`) for key metrics or titles.
-   Use bullet points (`-`) for lists.
-   Reference images by including the image path directly in your response. For example: `temp_plot_images/a-valid-uuid-here.png`
-   Ensure all relevant image paths are included at the end of your text response.

### Example of Strict Output Format:
---
I have generated a sales report based on the uploaded data.

## Overall Performance Summary
**Total Sales:** [Total Sales Value from your analysis]
**Total Profit:** [Total Profit Value from your analysis]

## Sales Breakdown
### Top 5 Products by Sales
-   **Product 1:** [Sales Value]
-   **Product 2:** [Sales Value]

Here is a visualization of the sales data:
{PLOT_DIRECTORY}/a-valid-uuid-here.png

---
Do not attempt to generate or execute arbitrary Python code directly. You must rely exclusively on the provided tools to interact with the CSV data.

Start the conversation by politely asking the user to upload a CSV file.
"""

# --- Streamlit UI Components and State Management ---
st.set_page_config(page_title="Data Analyst Assistant", layout="wide")
st.title("📊 CSV Data Analyst Assistant")
st.info("I am a helpful AI assistant ready to analyze your CSV files. Please upload a file to get started!")

# --- Async Agent Initialization ---
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

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
        llm = init_chat_model(
            "google_genai:gemini-2.5-flash"
        )
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
        for filename in os.listdir(PLOT_DIRECTORY):
            file_path = os.path.join(PLOT_DIRECTORY, filename)
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
                display_message_content(message.content)

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
                            config={"configurable": {"thread_id": "100"}}
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
                            config={"configurable": {"thread_id": "100", "recursion_limit": 50}}
                        )
                    )
                    
                    # The response is an AIMessage with content and tool_calls
                    agent_message = response['messages'][-1]
                    display_message_content(agent_message.content)
                    st.session_state.chat_history.append(agent_message)
                    
                except Exception as e:
                    st.error(f"An error occurred during agent invocation: {e}")
                    logger.error(f"Agent invocation error: {e}", exc_info=True)
else:
    st.warning("Agent initialization failed. Please check the console for details.")