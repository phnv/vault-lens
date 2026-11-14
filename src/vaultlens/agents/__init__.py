"""LLM layer and LangGraph search tiers (ADR-0009 section 6.8–6.11).

Only place (with `llm/`) that imports `langchain_*` or `langgraph`.
Never imported by core search or models.
"""
