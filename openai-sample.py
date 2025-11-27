import os
import sys
import json
from openai import OpenAI

# ANSI color codes
RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

def main():
    # Configuration - defaults to local server but can be overridden
    # Common local servers: http://localhost:8000/v1, http://localhost:1234/v1
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:18000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY") # Many local servers don't require a real key
    # If the server requires a specific model name, set it here or via env var
    model = os.getenv("OPENAI_MODEL", "Qwen3-4B-Instruct-2507-4bit")

    print(f"Connecting to OpenAI-compatible API at: {base_url}")
    print(f"Using model: {model}")
    print("Type 'quit' or 'exit' to end the chat.")
    print("Available tools: add_numbers (e.g., 'What is 5 + 3?')")
    print("(You can set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL env vars to configure)")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    messages = []

    while True:
        try:
            user_input = input(BLUE + "\nUser: " + RESET)
        except KeyboardInterrupt:
            print(RED + "\nExiting..." + RESET)
            break

        if user_input.lower() in ['quit', 'exit']:
            break

        messages.append({"role": "user", "content": user_input})

        print(GREEN + "Assistant: " + RESET, end="", flush=True)
        
        full_response = ""
        tool_calls_dict = {}
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                tools=tools,
                tool_choice="auto"
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_response += delta.content
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.index is not None:
                            idx = tool_call_delta.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            tc = tool_calls_dict[idx]
                            if tool_call_delta.id:
                                tc["id"] = tool_call_delta.id
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tc["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tc["function"]["arguments"] += tool_call_delta.function.arguments
            
            print() # Newline
            
            # Sort tool calls by index
            tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
            
            # Debug: indicate if tool calls were detected
            if tool_calls:
                print(YELLOW + f"\n[DEBUG] Tool calls detected: {len(tool_calls)}" + RESET)
                for i, tc in enumerate(tool_calls):
                    print(YELLOW + f"  [{i}] {tc['function']['name']}({tc['function']['arguments']})" + RESET)
            
            # Add the assistant's message
            assistant_message = {"role": "assistant", "content": full_response}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            # Handle tool calls
            if tool_calls:
                print(YELLOW + "\n[TOOL EXECUTION]" + RESET)
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    
                    print(YELLOW + f"Executing: {function_name}({function_args})" + RESET)
                    if function_name == "add_numbers":
                        result = add_numbers(**function_args)
                        print(YELLOW + f"Result: {result}" + RESET)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": str(result)
                        })
                    else:
                        print(RED + f"Unknown tool: {function_name}" + RESET)
                
                # Make another call with tool results
                print(GREEN + "\nAssistant (with tool results): " + RESET, end="", flush=True)
                tool_stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                tool_response = ""
                for chunk in tool_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        tool_response += content
                print()
                messages.append({"role": "assistant", "content": tool_response})

        except Exception as e:
            print(RED + f"\nError during generation: {e}" + RESET)

if __name__ == "__main__":
    main()

