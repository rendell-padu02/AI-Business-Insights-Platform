# src/ai_agent.py

import os
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent


# ---------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------
def _build_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",  # Groq model
        temperature=0.1,
    )
    return llm


def _build_pandas_agent(df: pd.DataFrame, system_prompt: str, max_iterations: int):
    """
    Helper to create a single-DataFrame pandas agent.
    In the Python REPL, the dataframe is available as `df`.
    """
    llm = _build_llm()

    # NOTE: we keep this compatible with older langchain_experimental:
    #  - no agent_type
    #  - no agent_executor_kwargs / early_stopping_method
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=system_prompt,
        max_iterations=max_iterations,
    )
    return agent


# ---------------------------------------------------------------------
# Public single-DF agents
# ---------------------------------------------------------------------
def build_superstore_agent(superstore: pd.DataFrame):
    """
    Agent that ONLY knows about the Superstore dataframe.
    """
    system_prompt = """
You are an analytics assistant working with a single pandas DataFrame named `df`.

This DataFrame is the **Superstore sales data** and includes columns like:
- Order Date, Ship Date, Sales, Profit, Discount, Quantity, Category, Sub-Category,
  Segment, Region, State, City, etc.

VERY IMPORTANT:
- Use pandas code on `df` only.
- Do NOT invent data; always compute from `df`.
- Use at most a few (1–3) calls to the python_repl_ast tool.
- Once you have computed the answer, you MUST give a concise business-focused
  explanation and STOP. Do not keep re-running the same code.
- When you use python_repl_ast, the Action Input MUST be pure Python code
  (no backticks, no quotes around the whole snippet).
""".strip()

    return _build_pandas_agent(superstore, system_prompt, max_iterations=8)


def build_churn_agent(churn: pd.DataFrame):
    """
    Agent that ONLY knows about the Telco churn dataframe.
    """
    system_prompt = """
You are an analytics assistant working with a single pandas DataFrame named `df`
containing the **Telco churn dataset**.

IMPORTANT RULES YOU MUST FOLLOW:

1. The column 'Churn Label' contains string values 'Yes' and 'No'.
   - Always convert it using: df['Churn Flag'] = (df['Churn Label'] == 'Yes').astype(int)
   - Use 'Churn Flag' for any numeric analysis (correlation, churn rate, grouping).

2. When computing churn rate:
   - NEVER divide two entire DataFrames.
   - ALWAYS divide Series of equal length, like:
       churn_counts['Churn Count'] / totals['Total Count']

3. NEVER try to compute correlations directly on string columns.
   - First filter numeric columns:
       num_df = df.select_dtypes(include=['int64','float64'])

4. Use AT MOST **2 python_repl_ast** tool calls.
   After the computation, STOP and give a concise business explanation.

5. NEVER retry the same failing code again.
   If a method fails, adjust the logic instead of repeating it.

Your answers must always:
- Use only valid pandas operations
- Be concise and business-focused
""".strip()

    return _build_pandas_agent(churn, system_prompt, max_iterations=8)


# ---------------------------------------------------------------------
# Dual Agent Router
# ---------------------------------------------------------------------
class DualPandasAgent:
    """
    Wraps two pandas agents:

      - superstore_agent: for Superstore sales data
      - churn_agent:      for Telco churn data

    .run(question) will:
      - Route to the most relevant agent (superstore / churn)
      - If the question clearly touches BOTH, it will query both
        and return a combined answer.
    """

    def __init__(self, superstore_agent, churn_agent):
        self.superstore_agent = superstore_agent
        self.churn_agent = churn_agent

    # Simple keyword-based router
    def _decide_target(self, question: str) -> str:
        q = question.lower()

        super_keywords = [
            "order", "profit", "sales", "discount", "category", "segment",
            "shipment", "ship", "region", "superstore", "customer segment",
        ]
        churn_keywords = [
            "churn", "contract", "tenure", "monthly charges",
            "payment method", "telco", "phone", "internet service",
        ]

        use_super = any(k in q for k in super_keywords)
        use_churn = any(k in q for k in churn_keywords)

        if use_super and use_churn:
            return "both"
        elif use_churn:
            return "churn"
        elif use_super:
            return "super"
        else:
            # Default: assume Superstore if it's ambiguous
            return "super"

    def _call_agent(self, agent, question: str) -> str:
        """
        Always call the underlying AgentExecutor via .invoke and unwrap "output".
        This avoids deprecated .run() AND avoids recursion on our own object.
        """
        result = agent.invoke({"input": question})

        # Newer AgentExecutors return dict with "output"
        if isinstance(result, dict) and "output" in result:
            result = result["output"]

        # Fallback: sometimes it's just a string
        if isinstance(result, str) and "Agent stopped due to iteration limit" in result:
            # At least give a friendlier message to the UI
            return (
                "The agent reached its internal iteration limit before finishing. "
                "Try asking a simpler or more specific question (for example, "
                "\"Which region has the highest total profit?\" instead of combining "
                "too many tasks in one prompt)."
            )

        return result

    def run(self, question: str) -> str:
        target = self._decide_target(question)

        if target == "super":
            return self._call_agent(self.superstore_agent, question)

        if target == "churn":
            return self._call_agent(self.churn_agent, question)

        # target == "both": ask each agent to answer from its own perspective
        super_q = (
            question
            + " Answer ONLY from the perspective of the Superstore sales dataset."
        )
        churn_q = (
            question
            + " Answer ONLY from the perspective of the Telco churn dataset."
        )

        super_answer = self._call_agent(self.superstore_agent, super_q)
        churn_answer = self._call_agent(self.churn_agent, churn_q)

        combined = (
            "### 🛒 Superstore perspective\n"
            + str(super_answer)
            + "\n\n---\n\n"
            + "### 📞 Churn perspective\n"
            + str(churn_answer)
        )
        return combined

    # Optional: small helper so you *can* use invoke() from outside.
    def invoke(self, input):
        if isinstance(input, dict):
            question = input.get("input", "")
        else:
            question = str(input)
        return {"output": self.run(question)}


# ---------------------------------------------------------------------
# Main "double agent" builder used by Streamlit
# ---------------------------------------------------------------------
def build_ai_agent(superstore: pd.DataFrame, churn: pd.DataFrame):
    """
    Build a dual agent that knows about BOTH:

      - Superstore dataframe
      - Churn dataframe

    Returned object has a `.run(question: str)` method and an `.invoke(...)`
    method that you can call from your Streamlit code.
    """
    superstore_agent = build_superstore_agent(superstore)
    churn_agent = build_churn_agent(churn)
    return DualPandasAgent(superstore_agent, churn_agent)
