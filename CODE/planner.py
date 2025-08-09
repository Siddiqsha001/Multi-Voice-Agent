import os
import traceback
from dotenv import load_dotenv
from memory import agent_memories, llm, save_to_memory
from memory_store import store_memory
from langchain.schema import HumanMessage
from state import AgentState
from websearch import search_web
from typing import Optional
from prompts import PLANNER_PROMPT


load_dotenv()
def analyze_topic_type(user_input: str)->str:
    try:
        messages=[
            HumanMessage(content=f"""Analyze this input and determine if it's career, education, or technical related.
Input: {user_input} Classify as one of: career, education, technical
Response format: Just the classification word in lowercase.""")
        ]
        result=llm.invoke(messages)
        response=result.content.strip().lower()
        return response if response in {"career", "education", "technical"} else "general"
    except Exception as e:
        print(f"Error analyzing topic type: {str(e)}")
        return "general"

def generate_expert_response(user_input: str, web_context: str, conversation_history: str, relevant_history: str) -> str:
    """Generate an expert response based on context and history"""
    try:
        chain_input={
            "user_input": user_input,
            "web_context": web_context,
            "history": conversation_history,
            "relevant_history": relevant_history
        }
        messages=[
            HumanMessage(content=PLANNER_PROMPT.format(**chain_input))
        ]
        result=llm.invoke(messages)
        return result.content.strip()
    except Exception as e:
        print(f"Error generating expert response: {str(e)}")
        return "I need more information to provide expert guidance. Could you provide more details?"

def calculate_expert_confidence(user_input: str, topic_type: str, web_results: str) -> float:
    confidence=0.0
    topic_confidence={
        "career": 0.3,
        "education": 0.3,
        "technical": 0.4
    }
    confidence+=topic_confidence.get(topic_type, 0.0)
    if web_results and len(web_results.strip())>0:
        confidence+=0.3
    expert_indicators=[
        "how", "compare", "difference", "best", "recommend", "explain",
        "analyze", "steps", "guide", "learn", "implement"
    ]
    if any(indicator in user_input.lower() for indicator in expert_indicators):
        confidence+=0.3
    return min(confidence, 1.0)

def planner_node(state: AgentState) -> AgentState:
    user_input=state.user_input
    new_state=AgentState(**state.dict())

    try:
        topic_type=analyze_topic_type(user_input)
        new_state.topic_type=topic_type
        search_queries={
            "career":f"career planning methodology {user_input} expert advice steps timeline",
            "education":f"learning path methodology {user_input} expert guidance timeline",
            "technical":f"technical implementation guide {user_input} best practices timeline",
            "general":f"step by step guide {user_input} methodology timeline"
        }
        search_query=search_queries.get(topic_type, search_queries["general"])
        search_result=search_web(search_query)
        new_state.web_results=search_result

        #memory retrieval with context
        memory=agent_memories["planner"]
        memory_vars=memory.load_memory_variables({memory.input_key: user_input})
        conversation_history=memory_vars.get("history", "")
        relevant_history=memory_vars.get("relevant_history", "")

        response=generate_expert_response(
            user_input, 
            search_result,
            conversation_history,
            relevant_history
        )
        confidence=calculate_expert_confidence(user_input,topic_type,search_result)

        new_state.planner_response=response
        new_state.planner_confidence=confidence

        #storing the interaction in memory
        save_to_memory(memory,user_input,response)
        store_memory(user_input,response,"planner")
        return new_state
    except Exception as e:
        print(f"Error in planner_node:{str(e)}\n{traceback.format_exc()}")
        new_state.planner_response="I encountered an error while processing your request."
        new_state.planner_confidence=0.0
        return new_state
