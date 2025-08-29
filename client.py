from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def main():
    # Initialize MCP client with CSV analyzer server
    client = MultiServerMCPClient(
        {
            "csv_analyzer": {
                "command": "python3",
                "args": ["tools/csv_tools.py"],  # Ensure correct path to your MCP server
                "transport": "stdio",
            }
        }
    )
    
    # Get available tools from the MCP server
    tools = await client.get_tools()
    available_tools = [tool.name for tool in tools]
    print(f"Available tools: {available_tools}")
    
    # Initialize the language model (make sure you have GOOGLE_API_KEY in .env)
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    
    # Create the ReAct agent with tools
    agent = create_react_agent(model, tools)
    
    # # Example 1: List uploaded CSV files
    # print("\n" + "="*50)
    # print("LISTING UPLOADED CSV FILES")
    # print("="*50)
    
    # list_response = await agent.ainvoke({
    #     "messages": [{
    #         "role": "user", 
    #         "content": "List all CSV files that have been uploaded to the system"
    #     }]
    # })
    # print("Response:", list_response['messages'][-1].content)
    
    # Example 2: Get a preview of a CSV file
    # print("\n" + "="*50)
    # print("CSV FILE PREVIEW")
    # print("="*50)
    
    # preview_response = await agent.ainvoke({
    #     "messages": [{
    #         "role": "user", 
    #         "content": "Show me a preview of the first CSV file you find - just the first 3 rows"
    #     }]
    # })
    # print("Response:", preview_response['messages'][-1].content)
    
    # # Example 3: Full analysis of a CSV file
    # print("\n" + "="*50)
    # print("FULL CSV ANALYSIS")
    # print("="*50)
    
    # analysis_response = await agent.ainvoke({
    #     "messages": [{
    #         "role": "user", 
    #         "content": """Analyze the first CSV file you can find and provide me with:
    #         1. Basic dataset information (rows, columns, file size)
    #         2. Data quality summary (missing values, data types)
    #         3. Key insights about numeric columns (if any)
    #         4. Sample values from each column
            
    #         Present this in a clear, organized format."""
    #     }]
    # })
    # print("Response:", analysis_response['messages'][-1].content)
    
    # Example 4: Targeted analysis with specific requirements
    # print("\n" + "="*50)
    # print("TARGETED ANALYSIS")
    # print("="*50)
    
    # targeted_response = await agent.ainvoke({
    #     "messages": [{
    #         "role": "user",
    #         "content": """Find the CSV file that contains credit data. 
    #         Use the available tools to list files or preview them if necessary. 
    #         Then analyze the first 100 rows of that file:
    #         - Columns with most missing data
    #         - Numeric columns with most variation
    #         - Any data quality issues
    #         If there are more than 5 columns, just analyze the first 5."""
    #     }]
    # })
    # print("Response:", targeted_response['messages'][-1].content)
    
    # # Example 5: Interactive session - let user ask questions
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("You can now ask questions about your CSV files!")
    print("Type 'quit' to exit.")
    
    while True:
        system_prompt = f"""
        You are an intelligent agent that can use the following tools: {available_tools}.

        Step 1: Decide which tool(s) to use to answer the user question.
        Step 2: Write a step-by-step plan describing which tool to call and in what order.
        Step 3: Execute the plan to answer the user question.
        User question : 
        """
        user_question = input("\nYour question: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
            
        interactive_response = await agent.ainvoke({
            "messages": [{
                "role": "user", 
                "content": system_prompt + user_question
            }]
        })
        print("\nAgent:", interactive_response['messages'][-1].content)

# Additional utility function for focused analysis
async def analyze_specific_file(filename: str, question: str = None):
    """
    Analyze a specific CSV file with an optional custom question.
    
    Args:
        filename: Name of the CSV file to analyze
        question: Custom analysis question (optional)
    """
    client = MultiServerMCPClient({
        "csv_analyzer": {
            "command": "python3",
            "args": ["csv_analyzer_mcp.py"],
            "transport": "stdio",
        }
    })
    
    tools = await client.get_tools()
    model = init_chat_model("google_genai:gemini-2.0-flash-exp")
    agent = create_react_agent(model, tools)
    
    if question is None:
        question = f"Provide a comprehensive analysis of the file '{filename}' including data types, missing values, statistical summaries, and any data quality insights."
    
    response = await agent.ainvoke({
        "messages": [{
            "role": "user", 
            "content": f"Analyze the CSV file '{filename}'. {question}"
        }]
    })
    
    return response['messages'][-1].content

# Example usage of the utility function
async def demo_specific_analysis():
    """Demo the specific file analysis function"""
    # Replace 'your_file.csv' with an actual file in your temp_data_files directory
    result = await analyze_specific_file(
        "data.csv",  # Change this to your actual filename
        "What are the key characteristics of this dataset? Are there any data quality issues?"
    )
    print(result)

if __name__ == "__main__":
    # Run the main interactive demo
    asyncio.run(main())
    
    # Uncomment to run specific file analysis instead:
    # asyncio.run(demo_specific_analysis())