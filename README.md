# AI-Routed Dynamic Task Dispatch & Operations Workflow

## Overview
This repository contains a Proof of Concept (PoC) for an AI-driven multi-agent workflow designed to eliminate information silos and accelerate ticket resolution for agile startup teams. Built on **LangGraph** and powered by **OpenAI GPT-4o**, the system dynamically categorizes incoming user queries, routes them to specialized handler nodes, executes parallel data retrievals, and implements a robust Human-in-the-Loop (HITL) escalation mechanism.

## Architecture & Workflow

1. **Input Preprocessing (`normalize_input`)**: 
   Dynamically accepts both raw text strings and JSON payloads. It normalizes the input to extract the core `user_query` and `user_id` before entering the LLM pipeline.
   
2. **Intelligent Dispatch (`triage_agent`)**: 
   A dedicated routing node utilizes GPT-4o's structured output capabilities to classify the intent of the request (e.g., `refund`, `pricing`, `technical_issue`). 
   
3. **Conditional Routing & Parallel Execution (`handle_refund` / `handle_general`)**: 
   Based on the triage classification, the graph directs the flow to specific sub-processes.
   * *Cost-Optimization Strategy*: Instead of making redundant LLM calls for verification, the system executes **concurrent Python threads** to fetch heterogeneous context data (e.g., retrieving company policy and user purchase history simultaneously). This enriches the LLM's context window, maximizing deterministic accuracy while minimizing token costs.
   
4. **Resolution & Escalation (`escalation_agent`)**: 
   The support node determines the `is_resolved` status. If the AI lacks the authority or capability to resolve the issue, the state is conditionally routed to the Escalation Agent. In a production environment, this node acts as a breakpoint (`interrupt_before`), suspending the graph execution and queuing the ticket for asynchronous human expert review.

## Prerequisites
* Python 3.9+
* OpenAI API Key configured in your environment (`OPENAI_API_KEY`)

## Installation
Install the required dependencies:
```bash
pip install langgraph langchain-openai pydantic
```

## Future fix
1. Reduced Token Expenditure: Employs parallel deterministic data fetching over expensive multi-prompt self-consistency loops.

2. Deterministic Fallback: Ensures 100% human oversight for edge cases via state machine suspension, preventing AI hallucinations from causing customer service disasters.

3. Plug-and-Play Extensibility: New categories and API integrations (e.g., Stripe for refunds, Jira for technical issues) can be added by simply defining a new node and mapping a conditional edge.