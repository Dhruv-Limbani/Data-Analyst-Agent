from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
import asyncio
from dotenv import load_dotenv

load_dotenv()

class ConversationState:
    def __init__(self):
        self.awaiting_clarification = False
        self.last_system_prompt = None

async def generate_plan_and_clarification(llm, user_message, tools_info):
    """
    LLM generates both:
    - optional clarification
    - a system prompt describing what to do
    """
    prompt = f"""
You are an intelligent assistant. A user asked:
"{user_message}"

You have access to these tools: {tools_info}

Steps:
1. Identify the user's intent.
2. Determine any missing info needed to answer the question.
3. If info is missing, propose a clarifying question.
4. Generate a concise system prompt describing:
   - The intent
   - Which tools to call and in what order

Respond in JSON:
{{
  "clarification": "optional clarifying question, empty if confident",
  "system_prompt": "system prompt for yourself to execute the plan"
}}
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    import json
    try:
        data = json.loads(response.content)
        return data.get("clarification"), data.get("system_prompt")
    except Exception:
        return None, prompt

async def main():
    state = ConversationState()

    client = MultiServerMCPClient({
        "csv_analyzer": {"command": "python3", "args": ["tools/csv_tools.py"], "transport": "stdio"},
        "local_python_executor": {"command": "python3", "args": ["tools/local_python_executor.py"], "transport": "stdio"}
    })
    tools = await client.get_tools()
    tools_info = [{"name": t.name, "description": t.description, "args": t.args_schema} for t in tools]

    llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
    agent = create_react_agent(llm, tools)

    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Ask questions about your CSV files. Type 'quit' to exit.")

    while True:
        user_message = input("\nYour question: ")
        if user_message.lower() in ["quit", "exit", "q"]:
            break

        if state.awaiting_clarification:
            # User responded to clarification
            system_prompt = state.last_system_prompt + f"\nUser provided: {user_message}"
            state.awaiting_clarification = False
        else:
            # LLM decides intent & clarification
            clarification, system_prompt = await generate_plan_and_clarification(llm, user_message, tools_info)
            if clarification:
                print("\nAgent (clarifying):", clarification)
                state.awaiting_clarification = True
                state.last_system_prompt = system_prompt
                continue
            state.last_system_prompt = system_prompt

        # Execute LLM with planned system prompt
        interactive_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": system_prompt + "\nUser query: " + user_message}]
        })
        print("\nAgent:", interactive_response['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
