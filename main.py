# Dependencies: pip install langgraph langchain-openai pydantic
import json
import concurrent.futures
from typing import TypedDict, Any, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# 1. Define state dictionary structure, supporting both JSON and plain text inputs
class WorkflowState(TypedDict):
    raw_input: Any
    user_query: str
    user_id: str
    category: str
    context_data: Dict[str, Any]
    support_reply: str
    is_resolved: bool

# 2. Constrain LLM output format
class TriageResult(BaseModel):
    category: str = Field(description="Must be an enum value: refund, pricing, technical_issue, unknown")
    confidence: float = Field(description="Classification confidence score, range 0.0 - 1.0")

class SupportResult(BaseModel):
    is_resolved: bool = Field(description="Whether the issue is resolved (counts as resolved if rejected by policy, unless human intervention is strictly required)")
    reply: str = Field(description="Final text reply to the user")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Node: Input preprocessing (Pre-routing check)
def normalize_input(state: WorkflowState) -> WorkflowState:
    raw = state.get("raw_input", "")
    if isinstance(raw, dict):
        return {"user_query": raw.get("query", ""), "user_id": raw.get("user_id", "anonymous")}
    elif isinstance(raw, str):
        try: # Attempt to parse plain text as JSON
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {"user_query": parsed.get("query", ""), "user_id": parsed.get("user_id", "anonymous")}
        except json.JSONDecodeError:
            pass
        return {"user_query": raw, "user_id": "anonymous"}
    return {"user_query": str(raw), "user_id": "anonymous"}

# 4. Node: Triage Agent powered by GPT-4o
def triage_agent(state: WorkflowState) -> WorkflowState:
    prompt = f"Accurately classify the following user request:\nRequest: {state['user_query']}"
    result = llm.with_structured_output(TriageResult).invoke([HumanMessage(content=prompt)])
    return {"category": result.category}

# 5. Node: Specific processing workflows (Parallel task splitting)
def handle_refund(state: WorkflowState) -> WorkflowState:
    def fetch_refund_policy():
        return "Company Policy: Unconditional refunds are supported within 7 days of purchase."
        
    def fetch_user_history(uid):
        return f"System Record: User {uid} purchased a premium subscription 3 days ago."

    # Concurrently fetch heterogeneous data to replace repetitive LLM calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_policy = executor.submit(fetch_refund_policy)
        f_history = executor.submit(fetch_user_history, state["user_id"])
        context = {"policy": f_policy.result(), "history": f_history.result()}
    
    prompt = f"User Request: {state['user_query']}\nPolicy: {context['policy']}\nRecord: {context['history']}\nDetermine if the refund conditions are met and generate a formal reply."
    result = llm.with_structured_output(SupportResult).invoke([
        SystemMessage(content="You are an automated customer service expert, strictly following policies."), 
        HumanMessage(content=prompt)
    ])
    
    return {"context_data": context, "support_reply": result.reply, "is_resolved": result.is_resolved}

def handle_general(state: WorkflowState) -> WorkflowState:
    prompt = f"User Request: {state['user_query']}\nProvide preliminary answers or technical guidance."
    result = llm.with_structured_output(SupportResult).invoke([
        SystemMessage(content="You are a technical and business support expert."), 
        HumanMessage(content=prompt)
    ])
    return {"support_reply": result.reply, "is_resolved": result.is_resolved}

# 6. Node: Escalation and suspension
def escalation_agent(state: WorkflowState) -> WorkflowState:
    return {"support_reply": state.get("support_reply", "") + "\n\n[System Alert]: AI lacks privileges or failed to resolve the issue. The ticket has been routed to the human review queue."}

# 7. Define conditional routing edges
def route_triage(state: WorkflowState) -> str:
    return "handle_refund" if state["category"] == "refund" else "handle_general"

def route_support(state: WorkflowState) -> str:
    return END if state["is_resolved"] else "escalation_agent"

# 8. Assemble the state machine graph
workflow = StateGraph(WorkflowState)
workflow.add_node("normalize", normalize_input)
workflow.add_node("triage", triage_agent)
workflow.add_node("handle_refund", handle_refund)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalation_agent", escalation_agent)

workflow.add_edge(START, "normalize")
workflow.add_edge("normalize", "triage")
workflow.add_conditional_edges("triage", route_triage, {"handle_refund": "handle_refund", "handle_general": "handle_general"})
workflow.add_conditional_edges("handle_refund", route_support, {END: END, "escalation_agent": "escalation_agent"})
workflow.add_conditional_edges("handle_general", route_support, {END: END, "escalation_agent": "escalation_agent"})
workflow.add_edge("escalation_agent", END)

# To simulate real blocking (HITL), introduce a checkpointer and add interrupt_before=["escalation_agent"] during compilation
app = workflow.compile()