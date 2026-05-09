
## Start
  - [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview)
  - **Core benefits**
    - LangGraph provides low-level supporting infrastructure for any long-running, stateful workflow or agent. LangGraph does not abstract prompts or architecture, and provides the following central benefits:
    - Durable execution: Build agents that persist through failures and can run for extended periods, resuming from where they left off.
    - Human-in-the-loop: Incorporate human oversight by inspecting and modifying agent state at any point.
    - Comprehensive memory: Create stateful agents with both short-term working memory for ongoing reasoning and long-term memory across sessions.
    - Debugging with LangSmith: Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
    - Production-ready deployment: Deploy sophisticated agent systems confidently with scalable infrastructure designed to handle the unique challenges of stateful, long-running workflows.
  - **When building an agent with LangGraph**:
    - First break it apart into discrete steps called nodes.
    - Then, describe the different decisions and transitions from each of your nodes.
    - Finally, connect nodes together through a shared state that each node can read from and write to.
  - **Hello world**
    ```sh
    pip install langchain langchain_core langchain-anthropic langgraph langchain-openai
    ```
    ```py
    from langgraph.graph import StateGraph, MessagesState, START, END

    def mock_llm(state: MessagesState):
        return {"messages": [{"role": "ai", "content": "hello world"}]}

    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()

    graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
    ```
## Local model response
  ```json
  curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "What is 2 times 3?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "multiply",
        "description": "Multiply two integers",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
          },
          "required": ["a", "b"]
        }
      }
    }]
  }'

  {"choices":[{"finish_reason":"tool_calls","index":0,"message":{
    "role":"assistant","content":"I'll multiply 2 by 3 for you.",
    "reasoning_content":"The user is asking for the result of multiplying 2 by 3. I have access to a multiply function that can perform this calculation. The function requires two integer parameters: \"a\" and \"b\". In this case, a=2 and b=3.\n\nI should call the multiply function with these values.",
    "tool_calls":[
      {"type":"function","function":{"name":"multiply","arguments":"{\"a\":2,\"b\":3}"},"id":"hVgFwDIU7UfRrD86FcATmg85xZeTznPt"}
    ]}}],
    "created":1778228524,"model":"unsloth/GLM-4.7-Flash",
    "system_fingerprint":"b8985-27aef3dd9","object":"chat.completion",
    "usage":{"completion_tokens":92,"prompt_tokens":174,"total_tokens":266,"prompt_tokens_details":{"cached_tokens":0}},
    "id":"chatcmpl-j7rXfpciWhfEaAgYIw0fhEFP8OdcL2Jb",
    "timings":{"cache_n":0,"prompt_n":174,"prompt_ms":241.661,"prompt_per_token_ms":1.3888563218390804,"prompt_per_second":720.0168831545017,
      "predicted_n":92,"predicted_ms":894.039,"predicted_per_token_ms":9.717815217391305,"predicted_per_second":102.9037883134852
    }
  }
  ```
## Basic Graph API Usage
  - basic graph
    ```
                    +--------done------> __end__
                    |
    __start__ --> llm_call --continue--> tool_node -+
                    ^                               |
                    +-------------------------------+
    ```
  - **Define model**
    ```py
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
        base_url="http://localhost:8080/v1",        # llama-server endpoint
        api_key="not-needed",                       # required by SDK, ignored by server
        temperature=0.6,
    )
    ```
  - **Define tools and bind**
    ```py
    # Define tools, DOC STRING IS ESSENTIAL
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a * b


    @tool
    def add(a: int, b: int) -> int:
        """Adds `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a + b


    @tool
    def divide(a: int, b: int) -> float:
        """Divide `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a / b


    # Augment the LLM with tools
    tools = [add, multiply, divide]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    ```
  - **Define state** The graph’s state is used to store the messages and the number of LLM calls.
    ```py
    from langchain.messages import AnyMessage
    from typing_extensions import TypedDict, Annotated
    import operator

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int
    ```
  - **Define model node** The model node is used to call the LLM and decide whether to call a tool or not.
    ```py
    from langchain.messages import SystemMessage


    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }
    ```
  - **Define tool node** The tool node is used to call the tools and return the results.
    ```py
    from langchain.messages import ToolMessage

    def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    ```
  - **Define end logic** The conditional edge function is used to route to the tool node or end based upon whether the LLM made a tool call.
    ```py
    from typing import Literal
    from langgraph.graph import StateGraph, START, END


    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END
    ```
  - **Build and compile the agent** The agent is built using the StateGraph class and compiled using the compile method.
    ```py
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    from IPython.display import Image, display
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    print(agent.get_graph(xray=True).draw_mermaid())
    # ---
    # config:
    #   flowchart:
    #     curve: linear
    # ---
    # graph TD;
    #         __start__([<p>__start__</p>]):::first
    #         llm_call(llm_call)
    #         tool_node(tool_node)
    #         __end__([<p>__end__</p>]):::last
    #         __start__ --> llm_call;
    #         llm_call -.-> __end__;
    #         llm_call -.-> tool_node;
    #         tool_node --> llm_call;
    #         classDef default fill:#f2f0ff,line-height:1.2
    #         classDef first fill-opacity:0
    #         classDef last fill:#bfb6fc

    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()

    # ================================ Human Message =================================
    #
    # Add 3 and 4.
    # ================================== Ai Message ==================================
    #
    # I'll add 3 and 4 for you using the add function.
    # Tool Calls:
    #   add (WHqoXx67acvQHBVoh6XrMXYubI90gOnV)
    #  Call ID: WHqoXx67acvQHBVoh6XrMXYubI90gOnV
    #   Args:
    #     a: 3
    #     b: 4
    # ================================= Tool Message =================================
    #
    # 7
    # ================================== Ai Message ==================================
    #
    # 3 + 4 = 7
    #
    ```
## Langgraph APP
  - **Setup with a template**
    ```sh
    pip install -U "langgraph-cli[inmem]"
    langgraph new ~/workspace/test_langgraph_app --template new-langgraph-project-python

    cd ~/workspace/test_langgraph_app/
    pip install -e .
    langgraph dev
    ```
  - **`langgraph.json` How langgraph dev finds the graph**
    ```json
    "graphs": { "agent": "./src/agent/graph.py:graph" }
    ```
    **Registering multiple graphs**
    ```json
    "graphs": {
      "agent": "./src/agent/graph.py:graph",
      "calculator": "./src/agent/calculator.py:graph"
    }
    ```
    That maps the assistant name agent → the graph symbol exported from `src/agent/graph.py`. So langgraph dev only requires two things:
    1. A module path that imports cleanly.
    2. A top-level variable (here named graph) that is a compiled StateGraph.
  - **`src/agent/graph.py` Define self graph**
    ```py
    from __future__ import annotations
    import operator
    from typing import Annotated, Literal
    from typing_extensions import TypedDict

    from langchain.tools import tool
    from langchain.messages import AnyMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END


    model = ChatOpenAI(
        model="unsloth/GLM-4.7-Flash",
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        temperature=0.6,
    )

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`."""
        return a * b

    @tool
    def add(a: int, b: int) -> int:
        """Adds `a` and `b`."""
        return a + b

    @tool
    def divide(a: int, b: int) -> float:
        """Divide `a` and `b`."""
        return a / b

    tools = [add, multiply, divide]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)


    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int


    def llm_call(state: MessagesState):
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }


    def tool_node(state: MessagesState):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            observation = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": result}


    def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END


    graph = (
        StateGraph(MessagesState)
        .add_node("llm_call", llm_call)
        .add_node("tool_node", tool_node)
        .add_edge(START, "llm_call")
        .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        .add_edge("tool_node", "llm_call")
        .compile(name="Calculator Agent")
    )
    ```
    - Key requirement: the module-level graph must be the compiled `result (.compile(...))`, not the builder.
    - Keep src/agent/__init__.py as-is - it re-exports graph so langgraph.json can find it.
    - Restart `langgraph dev`.
  - **Call**
    ```py
    from langgraph_sdk import get_client
    import asyncio

    client = get_client(url="http://localhost:2024")

    async def main():
        final_state = None
        async for chunk in client.runs.stream(
            None,
            "agent",
            input={"messages": [{"role": "human", "content": "What is (3 + 4) * 5 / 6?"}]},
            stream_mode="values",
        ):
            if chunk.event == "values":
                final_state = chunk.data            # overwrite each step
            elif chunk.event == "error":
                raise RuntimeError(chunk.data)
            print(f"Receiving new event of type: {chunk.event}...")
            print(chunk.data)
            print("\n\n")
        return final_state["messages"][-1]["content"]
    rr = asyncio.run(main())
    print(rr)
    # The result of (3 + 4) * 5 / 6 is **5.833333333333333**.
    #
    # Breaking it down:
    # - 3 + 4 = 7
    # - 7 × 5 = 35
    # - 35 ÷ 6 = 5.833333333333333 (or 5 5/6)
    ```
***

# A basic email agent
## Start with the process you want to automate
  - [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph#read-and-classify-nodes)
  - The agent should:
    - Read incoming customer emails
    - Classify them by urgency and topic
    - Search relevant documentation to answer questions
    - Draft appropriate responses
    - Escalate complex issues to human agents
    - Schedule follow-ups when needed
  - Example scenarios to handle:
    1. Simple product question: "How do I reset my password?"
    2. Bug report: "The export feature crashes when I select PDF format"
    3. Urgent billing issue: "I was charged twice for my subscription!"
    4. Feature request: "Can you add dark mode to the mobile app?"
    5. Complex technical issue: "Our API integration fails intermittently with 504 errors"​s
## Identify what each step needs to do
  - **LLM steps** When a step needs to understand, analyze, generate text, or make reasoning decisions:
    - **Classify intent**
      - Static context (prompt): Classification categories, urgency definitions, response format
      - Dynamic context (from state): Email content, sender information
      - Desired outcome: Structured classification that determines routing
    - **Draft reply**
      - Static context (prompt): Tone guidelines, company policies, response templates
      - Dynamic context (from state): Classification results, search results, customer history
      - Desired outcome: Professional email response ready for review​
  - **Data steps** When a step needs to retrieve information from external sources:
    - **Document search**
      - Parameters: Query built from intent and topic
      - Retry strategy: Yes, with exponential backoff for transient failures
      - Caching: Could cache common queries to reduce API calls
    - **Customer history lookup**
      - Parameters: Customer email or ID from state
      - Retry strategy: Yes, but with fallback to basic info if unavailable
      - Caching: Yes, with time-to-live to balance freshness and performance
  - **Action steps** When a step needs to perform an external action:
    - **Send reply**
      - When to execute node: After approval (human or automated)
      - Retry strategy: Yes, with exponential backoff for network issues
      - Should not cache: Each send is a unique action
    - **Bug track**
      - When to execute node: Always when intent is “bug”
      - Retry strategy: Yes, critical to not lose bug reports
      - Returns: Ticket ID to include in response
  - **User input steps** When a step needs human intervention:
    - **Human review node**
      - Context for decision: Original email, draft response, urgency, classification
      - Expected input format: Approval boolean plus optional edited response
      - When triggered: High urgency, complex issues, or quality concerns
## Design your state
  - **State** is the shared memory accessible to all nodes in your agent. Think of it as the notebook your agent uses to keep track of everything it learns and decides as it works through the process.
  - **Include in state** Does it need to persist across steps? If yes, it goes in state.
  - **Don't store** Can you derive it from other data? If yes, compute it when needed instead of storing it in state.
  - The state contains only raw data—no prompt templates, no formatted strings, no instructions. The classification output is stored as a single dictionary, straight from the LLM.
***

# Workflows
## LLMs and augmentations
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  """ Schema for structured output. Instead of replying as free text, reply as a JSON object with pydantic schema. """
  from pydantic import BaseModel, Field

  class SearchQuery(BaseModel):
      search_query: str = Field(None, description="Query that is optimized web search.")
      justification: str = Field(
          None, description="Why this query is relevant to the user's request."
      )


  # Augment the LLM with schema for structured output
  structured_llm = llm.with_structured_output(SearchQuery)

  # Invoke the augmented LLM
  output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
  print(output)
  # search_query='How does Calcium CT score relate to high cholesterol?' justification='The user is asking for the medical relationship between a specific diagnostic test (Calcium CT score) and a metabolic condition (high cholesterol). I need to explain the biological mechanism (atherosclerosis) and the clinical implication (risk assessment).'

  """ Define a tool """
  def multiply(a: int, b: int) -> int:
      return a * b

  # Augment the LLM with tools
  llm_with_tools = llm.bind_tools([multiply])

  # Invoke the LLM with input that triggers the tool call
  msg = llm_with_tools.invoke("What is 2 times 3?")

  # Get the tool call
  msg.tool_calls
  ```
## Prompt chaining
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  from typing_extensions import TypedDict
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display


  # Graph state
  class State(TypedDict):
      topic: str
      joke: str
      improved_joke: str
      final_joke: str


  # Nodes
  def generate_joke(state: State):
      """First LLM call to generate initial joke"""

      msg = llm.invoke(f"Write a short joke about {state['topic']}")
      return {"joke": msg.content}


  def check_punchline(state: State):
      """Gate function to check if the joke has a punchline"""

      # Simple check - does the joke contain "?" or "!"
      if "?" in state["joke"] or "!" in state["joke"]:
          return "Pass"
      return "Fail"


  def improve_joke(state: State):
      """Second LLM call to improve the joke"""

      msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
      return {"improved_joke": msg.content}


  def polish_joke(state: State):
      """Third LLM call for final polish"""
      msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
      return {"final_joke": msg.content}


  # Build workflow
  workflow = StateGraph(State)

  # Add nodes
  workflow.add_node("generate_joke", generate_joke)
  workflow.add_node("improve_joke", improve_joke)
  workflow.add_node("polish_joke", polish_joke)

  # Add edges to connect nodes
  workflow.add_edge(START, "generate_joke")
  workflow.add_conditional_edges(
      "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
  )
  workflow.add_edge("improve_joke", "polish_joke")
  workflow.add_edge("polish_joke", END)

  # Compile
  chain = workflow.compile()

  # Show workflow
  display(Image(chain.get_graph().draw_mermaid_png()))
  print(chain.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         generate_joke(generate_joke)
  #         improve_joke(improve_joke)
  #         polish_joke(polish_joke)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> generate_joke;
  #         generate_joke -. &nbsp;Pass&nbsp; .-> __end__;
  #         generate_joke -. &nbsp;Fail&nbsp; .-> improve_joke;
  #         improve_joke --> polish_joke;
  #         polish_joke --> __end__;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  state = chain.invoke({"topic": "cats"})
  print("Initial joke:")
  print(state["joke"])
  print("\n--- --- ---\n")
  if "improved_joke" in state:
      print("Improved joke:")
      print(state["improved_joke"])
      print("\n--- --- ---\n")

      print("Final joke:")
      print(state["final_joke"])
  else:
      print("Final joke:")
      print(state["joke"])
  ```
## Parallelization
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  # Graph state
  class State(TypedDict):
      topic: str
      joke: str
      story: str
      poem: str
      combined_output: str


  # Nodes
  def call_llm_1(state: State):
      """First LLM call to generate initial joke"""

      msg = llm.invoke(f"Write a joke about {state['topic']}")
      return {"joke": msg.content}


  def call_llm_2(state: State):
      """Second LLM call to generate story"""

      msg = llm.invoke(f"Write a story about {state['topic']}")
      return {"story": msg.content}


  def call_llm_3(state: State):
      """Third LLM call to generate poem"""

      msg = llm.invoke(f"Write a poem about {state['topic']}")
      return {"poem": msg.content}


  def aggregator(state: State):
      """Combine the joke, story and poem into a single output"""

      combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
      combined += f"STORY:\n{state['story']}\n\n"
      combined += f"JOKE:\n{state['joke']}\n\n"
      combined += f"POEM:\n{state['poem']}"
      return {"combined_output": combined}


  # Build workflow
  parallel_builder = StateGraph(State)

  # Add nodes
  parallel_builder.add_node("call_llm_1", call_llm_1)
  parallel_builder.add_node("call_llm_2", call_llm_2)
  parallel_builder.add_node("call_llm_3", call_llm_3)
  parallel_builder.add_node("aggregator", aggregator)

  # Add edges to connect nodes
  parallel_builder.add_edge(START, "call_llm_1")
  parallel_builder.add_edge(START, "call_llm_2")
  parallel_builder.add_edge(START, "call_llm_3")
  parallel_builder.add_edge("call_llm_1", "aggregator")
  parallel_builder.add_edge("call_llm_2", "aggregator")
  parallel_builder.add_edge("call_llm_3", "aggregator")
  parallel_builder.add_edge("aggregator", END)
  parallel_workflow = parallel_builder.compile()

  # Show workflow
  display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
  print(parallel_workflow.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         call_llm_1(call_llm_1)
  #         call_llm_2(call_llm_2)
  #         call_llm_3(call_llm_3)
  #         aggregator(aggregator)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> call_llm_1;
  #         __start__ --> call_llm_2;
  #         __start__ --> call_llm_3;
  #         call_llm_1 --> aggregator;
  #         call_llm_2 --> aggregator;
  #         call_llm_3 --> aggregator;
  #         aggregator --> __end__;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  state = parallel_workflow.invoke({"topic": "cats"})
  print(state["combined_output"])
  ```
## Routing
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  from typing_extensions import Literal
  from langchain.messages import HumanMessage, SystemMessage


  # Schema for structured output to use as routing logic
  class Route(BaseModel):
      step: Literal["poem", "story", "joke"] = Field(
          None, description="The next step in the routing process"
      )


  # Augment the LLM with schema for structured output
  router = llm.with_structured_output(Route)


  # State
  class State(TypedDict):
      input: str
      decision: str
      output: str


  # Nodes
  def llm_call_1(state: State):
      """Write a story"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_2(state: State):
      """Write a joke"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_3(state: State):
      """Write a poem"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_router(state: State):
      """Route the input to the appropriate node"""

      # Run the augmented LLM with structured output to serve as routing logic
      decision = router.invoke(
          [
              SystemMessage(
                  content="Route the input to story, joke, or poem based on the user's request."
              ),
              HumanMessage(content=state["input"]),
          ]
      )

      return {"decision": decision.step}


  # Conditional edge function to route to the appropriate node
  def route_decision(state: State):
      # Return the node name you want to visit next
      if state["decision"] == "story":
          return "llm_call_1"
      elif state["decision"] == "joke":
          return "llm_call_2"
      elif state["decision"] == "poem":
          return "llm_call_3"


  # Build workflow
  router_builder = StateGraph(State)

  # Add nodes
  router_builder.add_node("llm_call_1", llm_call_1)
  router_builder.add_node("llm_call_2", llm_call_2)
  router_builder.add_node("llm_call_3", llm_call_3)
  router_builder.add_node("llm_call_router", llm_call_router)

  # Add edges to connect nodes
  router_builder.add_edge(START, "llm_call_router")
  router_builder.add_conditional_edges(
      "llm_call_router",
      route_decision,
      {  # Name returned by route_decision : Name of next node to visit
          "llm_call_1": "llm_call_1",
          "llm_call_2": "llm_call_2",
          "llm_call_3": "llm_call_3",
      },
  )
  router_builder.add_edge("llm_call_1", END)
  router_builder.add_edge("llm_call_2", END)
  router_builder.add_edge("llm_call_3", END)

  # Compile workflow
  router_workflow = router_builder.compile()

  # Show the workflow
  display(Image(router_workflow.get_graph().draw_mermaid_png()))
  print(router_workflow.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         llm_call_1(llm_call_1)
  #         llm_call_2(llm_call_2)
  #         llm_call_3(llm_call_3)
  #         llm_call_router(llm_call_router)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> llm_call_router;
  #         llm_call_router -.-> llm_call_1;
  #         llm_call_router -.-> llm_call_2;
  #         llm_call_router -.-> llm_call_3;
  #         llm_call_1 --> __end__;
  #         llm_call_2 --> __end__;
  #         llm_call_3 --> __end__;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  state = router_workflow.invoke({"input": "Write me a joke about cats"})
  print(state["output"])
  ```
## Orchestrator-worker
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  from typing import Annotated, List
  import operator


  # Schema for structured output to use in planning
  class Section(BaseModel):
      name: str = Field(
          description="Name for this section of the report.",
      )
      description: str = Field(
          description="Brief overview of the main topics and concepts to be covered in this section.",
      )


  class Sections(BaseModel):
      sections: List[Section] = Field(
          description="Sections of the report.",
      )


  # Augment the LLM with schema for structured output
  planner = llm.with_structured_output(Sections)

  from langgraph.types import Send


  # Graph state
  class State(TypedDict):
      topic: str  # Report topic
      sections: list[Section]  # List of report sections
      completed_sections: Annotated[
          list, operator.add
      ]  # All workers write to this key in parallel
      final_report: str  # Final report


  # Worker state
  class WorkerState(TypedDict):
      section: Section
      completed_sections: Annotated[list, operator.add]


  # Nodes
  def orchestrator(state: State):
      """Orchestrator that generates a plan for the report"""

      # Generate queries
      report_sections = planner.invoke(
          [
              SystemMessage(content="Generate a plan for the report."),
              HumanMessage(content=f"Here is the report topic: {state['topic']}"),
          ]
      )

      return {"sections": report_sections.sections}


  def llm_call(state: WorkerState):
      """Worker writes a section of the report"""

      # Generate section
      section = llm.invoke(
          [
              SystemMessage(
                  content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
              ),
              HumanMessage(
                  content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
              ),
          ]
      )

      # Write the updated section to completed sections
      return {"completed_sections": [section.content]}


  def synthesizer(state: State):
      """Synthesize full report from sections"""

      # List of completed sections
      completed_sections = state["completed_sections"]

      # Format completed section to str to use as context for final sections
      completed_report_sections = "\n\n---\n\n".join(completed_sections)

      return {"final_report": completed_report_sections}


  # Conditional edge function to create llm_call workers that each write a section of the report
  def assign_workers(state: State):
      """Assign a worker to each section in the plan"""

      # Kick off section writing in parallel via Send() API
      return [Send("llm_call", {"section": s}) for s in state["sections"]]


  # Build workflow
  orchestrator_worker_builder = StateGraph(State)

  # Add the nodes
  orchestrator_worker_builder.add_node("orchestrator", orchestrator)
  orchestrator_worker_builder.add_node("llm_call", llm_call)
  orchestrator_worker_builder.add_node("synthesizer", synthesizer)

  # Add edges to connect nodes
  orchestrator_worker_builder.add_edge(START, "orchestrator")
  orchestrator_worker_builder.add_conditional_edges(
      "orchestrator", assign_workers, ["llm_call"]
  )
  orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
  orchestrator_worker_builder.add_edge("synthesizer", END)

  # Compile the workflow
  orchestrator_worker = orchestrator_worker_builder.compile()

  # Show the workflow
  display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))
  print(orchestrator_worker.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         orchestrator(orchestrator)
  #         llm_call(llm_call)
  #         synthesizer(synthesizer)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> orchestrator;
  #         llm_call --> synthesizer;
  #         orchestrator -.-> llm_call;
  #         synthesizer --> __end__;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})

  from IPython.display import Markdown
  print(state["final_report"])
  ```
## Evaluator-optimizer
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  # Graph state
  class State(TypedDict):
      joke: str
      topic: str
      feedback: str
      funny_or_not: str


  # Schema for structured output to use in evaluation
  class Feedback(BaseModel):
      grade: Literal["funny", "not funny"] = Field(
          description="Decide if the joke is funny or not.",
      )
      feedback: str = Field(
          description="If the joke is not funny, provide feedback on how to improve it.",
      )


  # Augment the LLM with schema for structured output
  evaluator = llm.with_structured_output(Feedback)


  # Nodes
  def llm_call_generator(state: State):
      """LLM generates a joke"""

      if state.get("feedback"):
          msg = llm.invoke(
              f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
          )
      else:
          msg = llm.invoke(f"Write a joke about {state['topic']}")
      return {"joke": msg.content}


  def llm_call_evaluator(state: State):
      """LLM evaluates the joke"""

      grade = evaluator.invoke(f"Grade the joke {state['joke']}")
      return {"funny_or_not": grade.grade, "feedback": grade.feedback}


  # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
  def route_joke(state: State):
      """Route back to joke generator or end based upon feedback from the evaluator"""

      if state["funny_or_not"] == "funny":
          return "Accepted"
      elif state["funny_or_not"] == "not funny":
          return "Rejected + Feedback"


  # Build workflow
  optimizer_builder = StateGraph(State)

  # Add the nodes
  optimizer_builder.add_node("llm_call_generator", llm_call_generator)
  optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

  # Add edges to connect nodes
  optimizer_builder.add_edge(START, "llm_call_generator")
  optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
  optimizer_builder.add_conditional_edges(
      "llm_call_evaluator",
      route_joke,
      {  # Name returned by route_joke : Name of next node to visit
          "Accepted": END,
          "Rejected + Feedback": "llm_call_generator",
      },
  )

  # Compile the workflow
  optimizer_workflow = optimizer_builder.compile()

  # Show the workflow
  display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
  print(optimizer_workflow.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         llm_call_generator(llm_call_generator)
  #         llm_call_evaluator(llm_call_evaluator)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> llm_call_generator;
  #         llm_call_evaluator -. &nbsp;Accepted&nbsp; .-> __end__;
  #         llm_call_evaluator -. &nbsp;Rejected + Feedback&nbsp; .-> llm_call_generator;
  #         llm_call_generator --> llm_call_evaluator;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  state = optimizer_workflow.invoke({"topic": "Cats"})
  print(state["joke"])
  ```
## Agents
  ```py
  from typing import Literal
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display

  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
      base_url="http://localhost:8080/v1",        # llama-server endpoint
      api_key="not-needed",                       # required by SDK, ignored by server
      temperature=0.6,
  )

  from langchain.tools import tool


  # Define tools
  @tool
  def multiply(a: int, b: int) -> int:
      """Multiply `a` and `b`.

      Args:
          a: First int
          b: Second int
      """
      return a * b


  @tool
  def add(a: int, b: int) -> int:
      """Adds `a` and `b`.

      Args:
          a: First int
          b: Second int
      """
      return a + b


  @tool
  def divide(a: int, b: int) -> float:
      """Divide `a` and `b`.

      Args:
          a: First int
          b: Second int
      """
      return a / b


  # Augment the LLM with tools
  tools = [add, multiply, divide]
  tools_by_name = {tool.name: tool for tool in tools}
  llm_with_tools = llm.bind_tools(tools)

  from langgraph.graph import MessagesState
  from langchain.messages import SystemMessage, HumanMessage, ToolMessage


  # Nodes
  def llm_call(state: MessagesState):
      """LLM decides whether to call a tool or not"""

      return {
          "messages": [
              llm_with_tools.invoke(
                  [
                      SystemMessage(
                          content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                      )
                  ]
                  + state["messages"]
              )
          ]
      }


  def tool_node(state: dict):
      """Performs the tool call"""

      result = []
      for tool_call in state["messages"][-1].tool_calls:
          tool = tools_by_name[tool_call["name"]]
          observation = tool.invoke(tool_call["args"])
          result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
      return {"messages": result}


  # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
  def should_continue(state: MessagesState) -> Literal["tool_node", END]:
      """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

      messages = state["messages"]
      last_message = messages[-1]

      # If the LLM makes a tool call, then perform an action
      if last_message.tool_calls:
          return "tool_node"

      # Otherwise, we stop (reply to the user)
      return END


  # Build workflow
  agent_builder = StateGraph(MessagesState)

  # Add nodes
  agent_builder.add_node("llm_call", llm_call)
  agent_builder.add_node("tool_node", tool_node)

  # Add edges to connect nodes
  agent_builder.add_edge(START, "llm_call")
  agent_builder.add_conditional_edges(
      "llm_call",
      should_continue,
      ["tool_node", END]
  )
  agent_builder.add_edge("tool_node", "llm_call")

  # Compile the agent
  agent = agent_builder.compile()

  # Show the agent
  display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
  print(agent.get_graph(xray=True).draw_mermaid())
  # ---
  # config:
  #   flowchart:
  #     curve: linear
  # ---
  # graph TD;
  #         __start__([<p>__start__</p>]):::first
  #         llm_call(llm_call)
  #         tool_node(tool_node)
  #         __end__([<p>__end__</p>]):::last
  #         __start__ --> llm_call;
  #         llm_call -.-> __end__;
  #         llm_call -.-> tool_node;
  #         tool_node --> llm_call;
  #         classDef default fill:#f2f0ff,line-height:1.2
  #         classDef first fill-opacity:0
  #         classDef last fill:#bfb6fc

  # Invoke
  messages = [HumanMessage(content="Add 3 and 4.")]
  messages = agent.invoke({"messages": messages})
  for m in messages["messages"]:
      m.pretty_print()
  ```
***

# Capabilities
## HITL With Interrupt
  ```py
  from typing import Literal, Optional, TypedDict

  from langgraph.checkpoint.memory import MemorySaver
  from langgraph.graph import StateGraph, START, END
  from langgraph.types import Command, interrupt


  class ApprovalState(TypedDict):
      action_details: str
      status: Optional[Literal["pending", "approved", "rejected"]]


  def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
      # Expose details so the caller can render them in a UI
      decision = interrupt({
          "question": "Approve this action?",
          "details": state["action_details"],
      })

      # Route to the appropriate node after resume
      return Command(goto="proceed" if decision else "cancel")


  def proceed_node(state: ApprovalState):
      return {"status": "approved"}


  def cancel_node(state: ApprovalState):
      return {"status": "rejected"}


  builder = StateGraph(ApprovalState)
  builder.add_node("approval", approval_node)
  builder.add_node("proceed", proceed_node)
  builder.add_node("cancel", cancel_node)
  builder.add_edge(START, "approval")
  builder.add_edge("proceed", END)
  builder.add_edge("cancel", END)

  # Use a more durable checkpointer in production
  checkpointer = MemorySaver()
  graph = builder.compile(checkpointer=checkpointer)

  config = {"configurable": {"thread_id": "approval-123"}}
  initial = graph.invoke(
      {"action_details": "Transfer $500", "status": "pending"},
      config=config,
  )
  print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

  # Resume with the decision; True routes to proceed, False to cancel
  resumed = graph.invoke(Command(resume=True), config=config)
  print(resumed["status"])  # -> "approved"
  ```
## Web search with tavily
  - **Install. Needs TAVILY_API_KEY from [Tavily API Platform](https://app.tavily.com/home)**
    ```sh
    pip install tavily langchain-tavily
    ```
  - **By tavily directly**
    ```py
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from tavily import TavilyClient   # pip install tavily-python; needs TAVILY_API_KEY

    llm = ChatOpenAI(
        model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
        base_url="http://localhost:8080/v1",        # llama-server endpoint
        api_key="not-needed",                       # required by SDK, ignored by server
        temperature=0.6,
    )

    class SearchQuery(BaseModel):
        search_query: str = Field(None, description="Query that is optimized web search.")
        justification: str = Field(
            None, description="Why this query is relevant to the user's request."
        )
    # Augment the LLM with schema for structured output
    structured_llm = llm.with_structured_output(SearchQuery)

    tavily = TavilyClient()

    q = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
    print("Searching for:", q.search_query)

    results = tavily.search(q.search_query, max_results=5)
    for r in results["results"]:
        print(r["title"], "—", r["url"])
        print(r["content"][:200], "...\n")
    ```
  - **By langchain_tavily with a _clean**
    ```py
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
        base_url="http://localhost:8080/v1",        # llama-server endpoint
        api_key="not-needed",                       # required by SDK, ignored by server
        temperature=0.6,
    )

    from langchain_tavily import TavilySearch
    from langchain.messages import HumanMessage, ToolMessage

    search_tool = TavilySearch(max_results=5)
    tools_by_name = {search_tool.name: search_tool}
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [HumanMessage("How does Calcium CT score relate to high cholesterol?")]

    def _clean(v):
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")     # peel off stray wrapping quotes
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v

    while True:
      resp = llm_with_tools.invoke(messages)
      messages.append(resp)

      if not resp.tool_calls:
          break                                 # LLM gave a final answer

      for call in resp.tool_calls:
          args = _clean(call["args"])
          print(f"\n[LLM called {call['name']} with: {args}]")
          result = tools_by_name[call["name"]].invoke(args)
          print(f"[got {len(result) if hasattr(result,'__len__') else '?'} chars/items back]")
          messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print("\n=== FINAL ===")
    print(resp.content)
    ```
  - **By tavily as a tool**
    ```py
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="unsloth/GLM-4.7-Flash",                              # name doesn't matter for llama.cpp
        base_url="http://localhost:8080/v1",        # llama-server endpoint
        api_key="not-needed",                       # required by SDK, ignored by server
        temperature=0.6,
    )

    from langchain.tools import tool
    from tavily import TavilyClient

    _tavily = TavilyClient()   # uses TAVILY_API_KEY

    @tool
    def web_search(query: str) -> str:
        """Search the web. Pass a single search query string."""
        res = _tavily.search(query, max_results=5, search_depth="advanced")
        return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in res["results"])

    llm_with_tools = llm.bind_tools([web_search])
    tools_by_name = {"web_search": web_search}

    messages = [HumanMessage("How does Calcium CT score relate to high cholesterol?")]

    while True:
      resp = llm_with_tools.invoke(messages)
      messages.append(resp)

      if not resp.tool_calls:
          break                                 # LLM gave a final answer

      for call in resp.tool_calls:
          print(f"\n[LLM called {call['name']} with: {call['args']}]")
          result = tools_by_name[call["name"]].invoke(call["args"])
          print(f"[got {len(result) if hasattr(result,'__len__') else '?'} chars/items back]")
          messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print("\n=== FINAL ===")
    print(resp.content)
    ```
