from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import asyncio
from dotenv import load_dotenv

load_dotenv()

config = {"configurable": {"thread_id":"1"}}

async def main():
    client = MultiServerMCPClient({
        "csv_analyzer": {"command": "python3", "args": ["tools/csv_tools.py"], "transport": "stdio"},
        "local_python_executor": {"command": "python3", "args": ["tools/local_python_executor.py"], "transport": "stdio"}
    })
    tools = await client.get_tools()

    llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

    agent = create_react_agent(llm, tools)

    chat_history = []

    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Ask questions about your CSV files. Type 'quit' to exit.")
    
    while True:
        user_message = input("\nYour question: ")
        if user_message.lower() in ["quit", "exit", "q"]:
            break

        # Append the user's message to the chat history
        chat_history.append(HumanMessage(content=user_message))

        # Invoke the agent with the entire chat history
        # The agent will use this history to maintain context
        interactive_response = await agent.ainvoke({
            "messages": chat_history
        }, config=config)
        
        # Get the agent's response
        agent_response_message = interactive_response['messages'][-1]
        
        # Append the agent's response to the chat history for the next turn
        chat_history.append(agent_response_message)
        
        print("\nAgent:", agent_response_message.content)

if __name__ == "__main__":
    asyncio.run(main())
