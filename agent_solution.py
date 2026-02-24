"""
================================================================================
  GenAI Agent: LangGraph + MCP with Reflection Loop & Human-in-the-Loop Gate
  Author: Satish
================================================================================

ARCHITECTURE OVERVIEW
─────────────────────
We build a stateful LangGraph agent that:
  1. Calls MCP tools (mock filesystem + database)
  2. REFLECTS on tool failures and retries with an alternative strategy
  3. PAUSES before destructive actions and waits for human approval

WHY LANGGRAPH?
  - Native support for cyclic graphs (essential for retry loops)
  - Built-in state management with TypedDict
  - interrupt_before / interrupt_after for HITL gates
  - Clean conditional edge routing = readable flow control
"""

# ── Standard Imports ──────────────────────────────────────────────────────────
import json
import time
from typing import TypedDict, Annotated, Literal
from enum import Enum

# ── LangGraph Imports ─────────────────────────────────────────────────────────
# WHY: LangGraph gives us graph-based orchestration with state persistence.
# StateGraph lets us define nodes (processing steps) and edges (transitions).
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # WHY: Enables HITL pause/resume

# ── LangChain / LLM Imports ───────────────────────────────────────────────────
# WHY: We use ChatOpenAI (swap for Claude/Gemini as needed) as the reasoning engine.
# Tool binding lets the LLM decide WHICH tool to call and with WHAT arguments.
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_RETRIES = 3          # Max reflection/retry attempts before giving up
DESTRUCTIVE_ACTIONS = {  # Actions that require human approval
    "delete_file",
    "update_database_record",
    "drop_table",
    "overwrite_file"
}


# ================================================================================
# SECTION 1: MOCK MCP SERVER
# ================================================================================
# WHY: In production, MCP (Model Context Protocol) provides a standardized 
# interface between the agent and tools/resources. Here we mock it to simulate:
#   - Successful reads
#   - File-not-found errors  
#   - Database records
#
# The mock deliberately introduces failures so we can showcase reflection.
# ================================================================================

class MockMCPServer:
    """
    Simulates an MCP server with a mock filesystem and database.
    WHY: MCP standardizes how LLMs interact with external tools — think of it
    as the "USB-C port" for AI tools. Our mock lets us test failure paths.
    """

    def __init__(self):
        # Mock filesystem — some files intentionally missing to trigger errors
        self._filesystem = {
            "/reports/q4_2024.csv": "revenue,2400000\ncosts,1800000\nprofit,600000",
            "/reports/q3_2024.csv": "revenue,2100000\ncosts,1700000\nprofit,400000",
            "/config/settings.json": '{"theme":"dark","version":"2.1"}',
            "/logs/app.log": "2024-01-01 INFO: App started\n2024-01-02 ERROR: Timeout",
        }

        # Mock database — employee records
        self._database = {
            "employees": [
                {"id": 1, "name": "Alice Chen", "role": "Engineer", "salary": 120000},
                {"id": 2, "name": "Bob Smith", "role": "Manager", "salary": 150000},
                {"id": 3, "name": "Carol White", "role": "Designer", "salary": 110000},
            ]
        }

    def read_file(self, path: str) -> dict:
        """Direct file access — fails if exact path not found."""
        if path in self._filesystem:
            return {"success": True, "content": self._filesystem[path], "path": path}
        return {
            "success": False,
            "error": "FILE_NOT_FOUND",
            "message": f"No file at path: {path}",
            "hint": "Try using search_files to locate the correct path."
        }

    def search_files(self, query: str, directory: str = "/") -> dict:
        """Fuzzy search across all files — the FALLBACK for failed direct reads."""
        matches = []
        query_lower = query.lower()
        for path, content in self._filesystem.items():
            if path.startswith(directory) and query_lower in path.lower():
                matches.append({"path": path, "preview": content[:100]})
        return {
            "success": True,
            "matches": matches,
            "count": len(matches)
        }

    def delete_file(self, path: str) -> dict:
        """DESTRUCTIVE: Permanently removes a file."""
        if path in self._filesystem:
            del self._filesystem[path]
            return {"success": True, "message": f"Deleted: {path}"}
        return {"success": False, "error": "FILE_NOT_FOUND", "message": f"Cannot delete: {path}"}

    def query_database(self, table: str, filters: dict = None) -> dict:
        """Read records from mock database."""
        if table not in self._database:
            return {"success": False, "error": "TABLE_NOT_FOUND", "message": f"No table: {table}"}
        records = self._database[table]
        if filters:
            for key, val in filters.items():
                records = [r for r in records if r.get(key) == val]
        return {"success": True, "records": records, "count": len(records)}

    def update_database_record(self, table: str, record_id: int, updates: dict) -> dict:
        """DESTRUCTIVE: Modifies an existing database record."""
        if table not in self._database:
            return {"success": False, "error": "TABLE_NOT_FOUND"}
        for record in self._database[table]:
            if record["id"] == record_id:
                record.update(updates)
                return {"success": True, "message": f"Updated record {record_id}", "record": record}
        return {"success": False, "error": "RECORD_NOT_FOUND", "message": f"No record with id={record_id}"}


# Global MCP server instance
mcp = MockMCPServer()


# ================================================================================
# SECTION 2: TOOL DEFINITIONS (LangChain @tool wrappers)
# ================================================================================
# WHY: We wrap MCP calls as LangChain tools so the LLM can invoke them via 
# function/tool calling. Each tool returns structured JSON the agent can reason over.
# The tool docstrings are CRITICAL — the LLM reads them to decide when to use each tool.
# ================================================================================

@tool
def read_file(path: str) -> str:
    """
    Read a file from the filesystem by its exact path.
    Returns file content on success, or an error with a hint if file not found.
    Use this first; fall back to search_files if this returns FILE_NOT_FOUND.
    
    Args:
        path: Absolute file path (e.g., '/reports/q4_2024.csv')
    """
    result = mcp.read_file(path)
    return json.dumps(result)


@tool
def search_files(query: str, directory: str = "/") -> str:
    """
    Search for files by name pattern. Use this when read_file fails with FILE_NOT_FOUND,
    or when you don't know the exact path. Returns matching file paths and previews.
    
    Args:
        query: Search term to match against file names (e.g., 'q4', 'settings')
        directory: Limit search to this directory prefix (default: search all)
    """
    result = mcp.search_files(query, directory)
    return json.dumps(result)


@tool
def delete_file(path: str) -> str:
    """
    ⚠️ DESTRUCTIVE ACTION: Permanently delete a file from the filesystem.
    This action CANNOT be undone. Requires human approval before execution.
    
    Args:
        path: Absolute path of file to delete
    """
    result = mcp.delete_file(path)
    return json.dumps(result)


@tool
def query_database(table: str, filters: str = "{}") -> str:
    """
    Query records from the database. Safe read operation, no approval needed.
    
    Args:
        table: Table name (e.g., 'employees')
        filters: JSON string of filter conditions (e.g., '{"role": "Engineer"}')
    """
    filter_dict = json.loads(filters) if filters else {}
    result = mcp.query_database(table, filter_dict)
    return json.dumps(result)


@tool
def update_database_record(table: str, record_id: int, updates: str) -> str:
    """
    ⚠️ DESTRUCTIVE ACTION: Update an existing database record.
    Modifies live data. Requires human approval before execution.
    
    Args:
        table: Table name (e.g., 'employees')
        record_id: Integer ID of the record to update
        updates: JSON string of field updates (e.g., '{"salary": 130000}')
    """
    updates_dict = json.loads(updates) if updates else {}
    result = mcp.update_database_record(table, record_id, updates_dict)
    return json.dumps(result)


# Tool registry — maps tool names to callable functions
TOOLS = [read_file, search_files, delete_file, query_database, update_database_record]
TOOL_MAP = {t.name: t for t in TOOLS}


# ================================================================================
# SECTION 3: AGENT STATE
# ================================================================================
# WHY: LangGraph agents are stateful. Every node reads from and writes to this 
# shared state object. Using TypedDict gives us type safety and clarity.
#
# Key design decisions:
#   - messages: Full conversation history (LLM context window)
#   - retry_count: Tracks reflection attempts to prevent infinite loops
#   - pending_approval: Stores a destructive action waiting for human sign-off
#   - last_error: The most recent tool error (for reflection reasoning)
# ================================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # WHY add_messages: auto-appends, doesn't overwrite
    retry_count: int                          # Guards against infinite reflection loops
    pending_approval: dict | None             # Holds destructive action awaiting HITL
    last_error: str | None                    # Context for reflection reasoning
    execution_log: list[str]                  # Audit trail for debugging


# ================================================================================
# SECTION 4: LLM SETUP
# ================================================================================
# WHY: We bind tools directly to the LLM so it can emit structured tool calls.
# The system prompt is carefully crafted to guide reflection behavior.
# ================================================================================

# WHY ChatOpenAI with temperature=0: Deterministic reasoning for tool selection.
# Swap model_name for "claude-3-5-sonnet" or "gemini-1.5-pro" as needed.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an intelligent file system and database agent with access to MCP tools.

IMPORTANT BEHAVIORAL RULES:
1. REFLECTION: If a tool returns an error (e.g., FILE_NOT_FOUND), DO NOT give up.
   - Analyze WHY it failed
   - Try an alternative: if read_file failed, use search_files to find the correct path
   - Explain your reasoning before retrying

2. DESTRUCTIVE ACTIONS: Tools marked with ⚠️ (delete_file, update_database_record)
   MUST be approved by a human before execution. The system will handle the approval gate.
   You may plan and call them — the system pauses for approval automatically.

3. CLARITY: If a user prompt is ambiguous, ask for clarification before acting.
   Better to ask than to perform the wrong action, especially destructive ones.

4. TRANSPARENCY: Always explain what you're doing and why, especially when retrying."""


# ================================================================================
# SECTION 5: GRAPH NODES
# ================================================================================
# WHY NODES: Each node is a discrete processing step. This separation makes the 
# agent's behavior debuggable, testable, and modifiable independently.
# ================================================================================

def agent_node(state: AgentState) -> AgentState:
    """
    MAIN REASONING NODE
    WHY: This is the LLM's "brain" — it reads the conversation history,
    decides what to do next, and either responds or emits a tool call.
    We inject a system prompt here to ensure consistent behavior.
    """
    messages = state["messages"]
    
    # Prepend system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    
    log = state.get("execution_log", [])
    log.append(f"[AGENT] Generated response. Tool calls: {len(response.tool_calls)}")
    
    return {
        "messages": [response],
        "execution_log": log
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """
    TOOL EXECUTION NODE
    WHY: We separate tool execution from reasoning so we can intercept calls,
    check for destructive actions BEFORE running them, and handle errors uniformly.
    
    Key logic:
    - If a tool is destructive → store it as pending_approval, DON'T execute yet
    - If tool fails → store error in last_error for the reflection node
    - If tool succeeds → clear retry counter and errors
    """
    last_message = state["messages"][-1]
    tool_results = []
    pending_approval = None
    last_error = None
    log = state.get("execution_log", [])

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # ── HITL CHECK: Is this a destructive action? ───────────────────────
        # WHY: We intercept BEFORE execution. Storing pending_approval triggers
        # the hitl_gate node which pauses the graph for human input.
        if tool_name in DESTRUCTIVE_ACTIONS:
            log.append(f"[HITL] Destructive action detected: {tool_name}({tool_args})")
            pending_approval = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_id": tool_call_id
            }
            # Return a placeholder message — actual execution happens post-approval
            tool_results.append(ToolMessage(
                content=json.dumps({
                    "status": "PENDING_APPROVAL",
                    "message": f"Action '{tool_name}' requires human approval. Waiting..."
                }),
                tool_call_id=tool_call_id
            ))
            continue

        # ── NORMAL TOOL EXECUTION ────────────────────────────────────────────
        try:
            tool_fn = TOOL_MAP[tool_name]
            result = tool_fn.invoke(tool_args)
            result_data = json.loads(result)

            log.append(f"[TOOL] {tool_name} → success={result_data.get('success', True)}")

            # Track errors for reflection
            if isinstance(result_data, dict) and not result_data.get("success", True):
                last_error = f"Tool '{tool_name}' failed: {result_data.get('error')} — {result_data.get('message')}"
                log.append(f"[ERROR] {last_error}")

            tool_results.append(ToolMessage(
                content=result,
                tool_call_id=tool_call_id
            ))

        except Exception as e:
            error_msg = f"Unexpected error in {tool_name}: {str(e)}"
            last_error = error_msg
            log.append(f"[ERROR] {error_msg}")
            tool_results.append(ToolMessage(
                content=json.dumps({"success": False, "error": "EXECUTION_ERROR", "message": str(e)}),
                tool_call_id=tool_call_id
            ))

    return {
        "messages": tool_results,
        "pending_approval": pending_approval,
        "last_error": last_error,
        "execution_log": log
    }


def reflection_node(state: AgentState) -> AgentState:
    """
    REFLECTION NODE
    WHY: When a tool fails, we don't just retry blindly. We ask the LLM to 
    REASON about the failure and devise an alternative strategy. This is the 
    "Reflection" pattern from agent design — the agent learns from its mistakes
    within a single run.
    
    This node injects a reflection prompt that forces the LLM to:
    1. Acknowledge what went wrong
    2. Hypothesize why
    3. Propose a concrete alternative action
    """
    last_error = state.get("last_error", "Unknown error")
    retry_count = state.get("retry_count", 0)
    log = state.get("execution_log", [])

    reflection_prompt = f"""
REFLECTION REQUIRED (Attempt {retry_count + 1}/{MAX_RETRIES})

A tool call just failed with this error:
  {last_error}

Please:
1. Explain in 1-2 sentences WHY this likely failed
2. Describe your ALTERNATIVE APPROACH
3. Execute the alternative approach immediately

Do not repeat the same action that just failed.
If read_file failed → use search_files to find the correct path first.
"""
    log.append(f"[REFLECT] Injecting reflection prompt. Retry {retry_count + 1}/{MAX_RETRIES}")

    return {
        "messages": [HumanMessage(content=reflection_prompt)],
        "retry_count": retry_count + 1,
        "last_error": None,  # Clear error after reflection is triggered
        "execution_log": log
    }


def hitl_gate_node(state: AgentState) -> AgentState:
    """
    HUMAN-IN-THE-LOOP GATE NODE
    WHY: Destructive actions (delete, update) are irreversible. Before executing,
    we PAUSE the graph and surface the pending action to a human operator.
    
    In production: This integrates with Slack, email, a web UI, or a CLI prompt.
    LangGraph's MemorySaver checkpoint allows the graph to be RESUMED after 
    approval without losing any state.
    
    The graph is interrupted BEFORE this node via interrupt_before=['hitl_gate'].
    The human calls graph.invoke() again with their approval decision injected
    into the state to resume execution.
    """
    pending = state.get("pending_approval")
    log = state.get("execution_log", [])

    if not pending:
        return state

    # ── PRESENT TO HUMAN ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ⚠️  HUMAN APPROVAL REQUIRED")
    print("═" * 60)
    print(f"  Action  : {pending['tool_name']}")
    print(f"  Args    : {json.dumps(pending['tool_args'], indent=2)}")
    print("═" * 60)
    
    # In a real system, this would be async — sent to Slack/UI and awaited.
    # Here we use CLI input for demonstration.
    decision = input("  Approve? (yes/no): ").strip().lower()

    if decision == "yes":
        log.append(f"[HITL] APPROVED: {pending['tool_name']}")
        
        # Execute the approved destructive action
        tool_fn = TOOL_MAP[pending["tool_name"]]
        result = tool_fn.invoke(pending["tool_args"])

        approval_result = ToolMessage(
            content=result,
            tool_call_id=pending["tool_call_id"]
        )
        return {
            "messages": [approval_result],
            "pending_approval": None,
            "execution_log": log
        }
    else:
        log.append(f"[HITL] REJECTED: {pending['tool_name']}")
        rejection_msg = ToolMessage(
            content=json.dumps({
                "success": False,
                "status": "REJECTED_BY_HUMAN",
                "message": "Human operator rejected this destructive action."
            }),
            tool_call_id=pending["tool_call_id"]
        )
        return {
            "messages": [rejection_msg],
            "pending_approval": None,
            "execution_log": log
        }


# ================================================================================
# SECTION 6: ROUTING LOGIC (Conditional Edges)
# ================================================================================
# WHY CONDITIONAL EDGES: The graph isn't linear. After each node, we decide 
# dynamically where to go next. This is what makes LangGraph powerful for agents.
# ================================================================================

def route_after_agent(state: AgentState) -> Literal["tool_executor", "end"]:
    """
    After the LLM responds: did it want to call a tool?
    WHY: If the LLM's last message has tool_calls, we execute them.
    Otherwise, the agent is done — return its final answer to the user.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"
    return "end"


def route_after_tools(state: AgentState) -> Literal["hitl_gate", "reflection", "agent", "end"]:
    """
    After tools execute: what happened?
    
    Priority order:
    1. HITL → a destructive action is pending approval
    2. REFLECTION → a tool failed AND we have retries left
    3. AGENT → normal flow, continue reasoning
    4. END → max retries exceeded
    
    WHY this priority: HITL must take precedence. Reflection only makes sense 
    if we haven't exhausted our retry budget (prevents infinite loops).
    """
    # 1. Destructive action needs human approval
    if state.get("pending_approval"):
        return "hitl_gate"

    # 2. A tool errored — reflect if we have budget
    if state.get("last_error"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "reflection"
        else:
            return "end"  # Exhausted retries

    # 3. Normal continuation
    return "agent"


# ================================================================================
# SECTION 7: GRAPH ASSEMBLY
# ================================================================================
# WHY MemorySaver: Enables checkpointing. When HITL pauses the graph, the entire
# state is saved. When the human resumes it, LangGraph restores from checkpoint.
# In production, swap for PostgresSaver or RedisSaver for persistence across processes.
# ================================================================================

def build_agent_graph() -> StateGraph:
    """
    Assembles the complete agent graph.
    
    Flow:
      agent → [has tool calls?]
                ├─ YES → tool_executor → [what happened?]
                │                          ├─ HITL pending → hitl_gate → agent
                │                          ├─ Error + retries → reflection → agent
                │                          └─ Normal → agent
                └─ NO → END
    """
    checkpointer = MemorySaver()
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("hitl_gate", hitl_gate_node)

    # Entry point
    graph.set_entry_point("agent")

    # Conditional routing after agent
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tool_executor": "tool_executor", "end": END}
    )

    # Conditional routing after tools
    graph.add_conditional_edges(
        "tool_executor",
        route_after_tools,
        {
            "hitl_gate": "hitl_gate",
            "reflection": "reflection",
            "agent": "agent",
            "end": END
        }
    )

    # After reflection → back to agent (with new reasoning context)
    graph.add_edge("reflection", "agent")

    # After HITL → back to agent (to process approval result)
    graph.add_edge("hitl_gate", "agent")

    return graph.compile(checkpointer=checkpointer)


# ================================================================================
# SECTION 8: MAIN — DEMO SCENARIOS
# ================================================================================

def run_demo():
    """
    Demonstrates three scenarios:
    1. REFLECTION: Agent asked for a file that doesn't exist → reflects → searches → finds it
    2. HITL: Agent asked to delete a file → pauses → waits for human approval
    3. COMBINED: Update a DB record (HITL) after looking it up (reflection on bad query)
    """
    agent = build_agent_graph()
    config = {"configurable": {"thread_id": "demo-session-1"}}

    print("\n" + "█" * 70)
    print("  SCENARIO 1: REFLECTION LOOP — File Not Found Recovery")
    print("█" * 70)
    
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="Please read the Q4 financial report from /reports/q4_finance_report.csv")],
            "retry_count": 0,
            "pending_approval": None,
            "last_error": None,
            "execution_log": []
        },
        config={"configurable": {"thread_id": "demo-1"}}
    )
    
    print("\n📋 EXECUTION LOG:")
    for entry in result["execution_log"]:
        print(f"  {entry}")
    print("\n🤖 FINAL RESPONSE:")
    print(result["messages"][-1].content)

    print("\n\n" + "█" * 70)
    print("  SCENARIO 2: HITL GATE — Destructive Action Approval")
    print("█" * 70)

    result2 = agent.invoke(
        {
            "messages": [HumanMessage(content="Delete the application log file at /logs/app.log")],
            "retry_count": 0,
            "pending_approval": None,
            "last_error": None,
            "execution_log": []
        },
        config={"configurable": {"thread_id": "demo-2"}}
    )

    print("\n📋 EXECUTION LOG:")
    for entry in result2["execution_log"]:
        print(f"  {entry}")

    print("\n\n" + "█" * 70)
    print("  SCENARIO 3: DATABASE UPDATE with HITL")
    print("█" * 70)

    result3 = agent.invoke(
        {
            "messages": [HumanMessage(content="Give Alice Chen a raise to $135,000 in the employees database.")],
            "retry_count": 0,
            "pending_approval": None,
            "last_error": None,
            "execution_log": []
        },
        config={"configurable": {"thread_id": "demo-3"}}
    )

    print("\n📋 EXECUTION LOG:")
    for entry in result3["execution_log"]:
        print(f"  {entry}")


if __name__ == "__main__":
    run_demo()
