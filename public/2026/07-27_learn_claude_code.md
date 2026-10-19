---
tags: [llm-agents, claude-code, agents]
summary: Learning how coding agents work — the tool-use loop and step-by-step agent sessions with the Anthropic SDK.
---
## S01
```py
# The entire secret of an AI coding agent in one pattern:
#     while stop_reason == "tool_use":
#         response = LLM(messages, tools)
#         execute tools
#         append results
#     +----------+      +-------+      +---------+
#     |   User   | ---> |  LLM  | ---> |  Tool   |
#     |  prompt  |      |       |      | execute |
#     +----------+      +---+---+      +----+----+
#                           ^               |
#                           |   tool_result |
#                           +---------------+
#                           (loop continues)

import subprocess
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))
MODEL = os.getenv('MODEL_ID')
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain"
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
}]
print(f"{MODEL = }\n{SYSTEM = }\n{TOOLS = }")
# MODEL = 'Qwen3.6-35B-A3B-MLX-8bit'
# SYSTEM = "You are a coding agent at ~/workspace/learning/claude_code. Use bash to solve tasks. Act, don't explain"
# TOOLS = [{'name': 'bash', 'description': 'Run a shell command.', 'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}]

query = 'Create a file called hello.py that prints "Hello, World!"'
messages = []
messages.append({"role": "user", "content": query})

""" Loop if response.stop_reason == 'tool_use'. Loop 1 """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [ToolUseBlock(id='call_5782e6e7', caller=None, input={'command': 'cat > hello.py << \'EOF\'\nprint("Hello, World!")\nEOF'}, name='bash', type='tool_use')]
# response.stop_reason = 'tool_use'

block = response.content[0]
print(f"{block.type = }\n{block.input = }")
# block.type = 'tool_use'
# block.input = {'command': 'cat > hello.py << \'EOF\'\nprint("Hello, World!")\nEOF'}
rr = subprocess.run(block.input['command'], shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
command_output = (rr.stdout + rr.stderr).strip() or "(no output)"
print(f"{command_output = }")
# command_output = '(no output)'
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": command_output}]})

""" Loop 2 """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [ToolUseBlock(id='call_41fd9869', caller=None, input={'command': 'python3 hello.py'}, name='bash', type='tool_use')]
# response.stop_reason = 'tool_use'

block = response.content[0]
print(f"{block.type = }\n{block.input = }")
# block.type = 'tool_use'
# block.input = {'command': 'python3 hello.py'}

rr = subprocess.run(block.input['command'], shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
command_output = (rr.stdout + rr.stderr).strip() or "(no output)"
print(f"{command_output = }")
# command_output = 'Hello, World!'
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": command_output}]})

""" Loop 3, end loop """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [TextBlock(citations=None, text='Done. The file `hello.py` has been created and prints "Hello, World!".', type='text', cache_control=None)]
# response.stop_reason = 'end_turn'

print("final output:", messages[-1]['content'][0].text)
# final output: Done. The file `hello.py` has been created and prints "Hello, World!".
```
## S05
```py
# The entire secret of an AI coding agent in one pattern:
#     while stop_reason == "tool_use":
#         response = LLM(messages, tools)
#         execute tools
#         append results
#     +----------+      +-------+      +---------+
#     |   User   | ---> |  LLM  | ---> |  Tool   |
#     |  prompt  |      |       |      | execute |
#     +----------+      +---+---+      +----+----+
#                           ^               |
#                           |   tool_result |
#                           +---------------+
#                           (loop continues)

import subprocess
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))
MODEL = os.getenv('MODEL_ID')
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain"

SYSTEM = f"You are a coding agent at {WORKDIR}. Before starting any multi-step task, use todo_write to plan your steps. Update status as you go."

safe_path = lambda p: (WORKDIR / p).resolve()
def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
}]
print(f"{MODEL = }\n{SYSTEM = }\n{TOOLS = }")
# MODEL = 'Qwen3.6-35B-A3B-MLX-8bit'
# SYSTEM = "You are a coding agent at ~/workspace/learning/claude_code. Use bash to solve tasks. Act, don't explain"
# TOOLS = [{'name': 'bash', 'description': 'Run a shell command.', 'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}]

query = 'Create a file called hello.py that prints "Hello, World!"'
messages = []
messages.append({"role": "user", "content": query})

""" Loop if response.stop_reason == 'tool_use'. Loop 1 """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [ToolUseBlock(id='call_5782e6e7', caller=None, input={'command': 'cat > hello.py << \'EOF\'\nprint("Hello, World!")\nEOF'}, name='bash', type='tool_use')]
# response.stop_reason = 'tool_use'

block = response.content[0]
print(f"{block.type = }\n{block.input = }")
# block.type = 'tool_use'
# block.input = {'command': 'cat > hello.py << \'EOF\'\nprint("Hello, World!")\nEOF'}
rr = subprocess.run(block.input['command'], shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
command_output = (rr.stdout + rr.stderr).strip() or "(no output)"
print(f"{command_output = }")
# command_output = '(no output)'
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": command_output}]})

""" Loop 2 """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [ToolUseBlock(id='call_41fd9869', caller=None, input={'command': 'python3 hello.py'}, name='bash', type='tool_use')]
# response.stop_reason = 'tool_use'

block = response.content[0]
print(f"{block.type = }\n{block.input = }")
# block.type = 'tool_use'
# block.input = {'command': 'python3 hello.py'}

rr = subprocess.run(block.input['command'], shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
command_output = (rr.stdout + rr.stderr).strip() or "(no output)"
print(f"{command_output = }")
# command_output = 'Hello, World!'
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": command_output}]})

""" Loop 3, end loop """
response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
messages.append({"role": "assistant", "content": response.content})
print(f"{response.content = }\n{response.stop_reason = }")
# response.content = [TextBlock(citations=None, text='Done. The file `hello.py` has been created and prints "Hello, World!".', type='text', cache_control=None)]
# response.stop_reason = 'end_turn'

print("final output:", messages[-1]['content'][0].text)
# final output: Done. The file `hello.py` has been created and prints "Hello, World!".
```
