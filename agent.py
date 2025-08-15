from typing import List
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import Ollama as OllamaLLM

from rag import RagPipeline

PDF_TOOL_DESCRIPTION = """
Use this tool to search the uploaded PDF(s) and retrieve the most relevant text chunks.
Provide a focused query. The tool returns raw snippets with page references.
Use these snippets as authoritative context when answering.
"""

class AgenticRAG:
    """
    Agent wrapper:
      - LLM: Ollama (e.g., llama3:latest)
      - Tools: search_pdf (only if PDFs ingested)
      - Behavior: ReAct (ZERO_SHOT_REACT_DESCRIPTION)
    """
    def __init__(self, model_name: str, temperature: float, rag: RagPipeline):
        self.rag = rag
        self.llm = OllamaLLM(model=model_name, temperature=temperature)

    def _build_tools(self) -> List[Tool]:
        tools: List[Tool] = []
        if self.rag and self.rag.has_docs():
            # PDF search tool
            def _pdf_search(query: str) -> str:
                results = self.rag.search(query, k=4)
                return "\n\n".join(results)

            tools.append(
                Tool(
                    name="search_pdf",
                    func=_pdf_search,
                    description=PDF_TOOL_DESCRIPTION.strip(),
                )
            )
        # If no docs, the agent will simply answer with the LLM (no tools).
        return tools

    def answer(self, question: str) -> str:
        tools = self._build_tools()

        # Strong, tool-aware prefix
        system_prefix = (
            "You are an expert assistant using ReAct. "
            "If a PDF search tool is available, first decide if the question can be answered from the PDF. "
            "If yes, call `search_pdf` with a focused query, read the snippets, and cite pages when relevant. "
            "If no relevant PDF content exists, answer directly. "
            "Be concise, accurate, and complete. If the PDF does not contain the answer, say so."
        )

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

        prompt = f"{system_prefix}\n\nUser question: {question}"
        result = agent.run(prompt)
        return result.strip()

