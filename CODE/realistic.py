import os
import traceback
from dotenv import load_dotenv
from memory import agent_memories, llm, save_to_memory
from langchain.schema.runnable import RunnablePassthrough
from prompts import REALIST_PROMPT
from websearch import search_web
from state import AgentState
load_dotenv()

def calculate_realist_confidence(user_input: str, web_results: str, topic_type: str) -> float:
    confidence=0.0
    topic_confidence={
        "career":0.3,
        "education":0.3,
        "technical":0.2
    }
    confidence+=topic_confidence.get(topic_type, 0.0)
    if web_results and len(web_results.strip()) > 0:
        confidence+=0.3
    practical_indicators=[
        "how","what steps","practical","realistic","actually","really",
        "implementation","specific","detail","consider","challenge"
    ]
    if any(indicator in user_input.lower() for indicator in practical_indicators):
        confidence+=0.4
    return min(confidence,1.0)
def realistic_node(state: AgentState) -> AgentState:
    user_input=state.user_input
    new_state=AgentState(**state.dict())
    try:
        search_queries={
            "career": f"latest statistics {user_input} job market data practical considerations",
            "education": f"practical advice {user_input} student experiences requirements costs",
            "technical": f"real-world usage {user_input} industry adoption challenges comparison"
        }
        topic=state.topic_type or "general"
        search_query=search_queries.get(topic, user_input)
        search_result=search_web(search_query)
        new_state.web_results=search_result

        #memory retrieval
        memory=agent_memories["realist"]
        memory_vars=memory.load_memory_variables({memory.input_key:user_input})
        
        #chain execution with context
        chain=({
                "user_input":RunnablePassthrough(),
                "web_context": lambda _:search_result or "",
                "history": lambda _:memory_vars.get("history", ""),
                "relevant_history": lambda _:memory_vars.get("relevant_history", "")
            }| REALIST_PROMPT| llm)

        response=chain.invoke(user_input).content.strip()

        if search_result in search_result:
            response+=f"\n\nSources : \n{search_result}"

        confidence=calculate_realist_confidence(user_input, search_result, state.topic_type)
        new_state.update_agent_response("realist", response,confidence)

        #save to memory
        save_to_memory("realist",user_input,response)
    except Exception as e:
        print(f"Realist error: {str(e)}")
        traceback.print_exc()
        new_state.update_agent_response(
            "realist",
            "I'm having trouble processing that right now.",
            0.1
        )
    return new_state
