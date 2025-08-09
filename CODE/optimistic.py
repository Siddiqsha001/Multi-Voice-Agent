import os
import traceback 
from dotenv import load_dotenv
from memory import agent_memories,llm,save_to_memory
from langchain.schema.runnable import RunnablePassthrough
from prompts import OPTIMIST_PROMPT
from websearch import search_web
from state import AgentState

load_dotenv

def calculate_optimist_confidence(user_input:str,topic_type:str)->float:
    confidence=0.0
    topic_confidence={
        "career":0.3,
        "education":0.3,
        "technical":0.2
    }
    confidence+=topic_confidence.get(topic_type,0.0)
    opportunity_indicators=[
        "want","hope","dream","future","better","improve","learn",
        "grow","opportunity","excited","interested","passion","goal"
    ]
    if any(indicator in user_input.lower() for indicator in opportunity_indicators):
        confidence+=0.4

    success_indicators=[
        "success","achieve","potential","possible","can","will",
        "progress","advance","develop","master","excel"
    ]
    if any(indicator in user_input.lower() for indicator in success_indicators):
        confidence+=0.3

    return min(confidence,1.0)

def optimistic_node(state:AgentState)->AgentState:
    user_input=state.user_input
    new_state=AgentState(**state.dict())

    try:
        #web search
        search_queries = {
            "career": f"success stories {user_input} positive outcomes career growth",
            "education": f"benefits advantages {user_input} student success stories",
            "technical": f"exciting developments {user_input} future potential innovations"
        }

        topic=state.topic_type or "general"
        search_query=search_queries.get(topic,user_input)
        search_result=search_web(search_query)
        new_state.web_results=search_result

        #memory retrieval
        memory=agent_memories["optimist"]
        memory_vars=memory.load_memory_variables({memory.input_key: user_input})
        
        #execute chain 
        chain=(OPTIMIST_PROMPT |llm |(lambda x: x.content if hasattr(x, "content") else str(x))
        )
        chain_input={
            "user_input": user_input,
            "web_context": search_result or "",
            "history": memory_vars.get("history", ""),
            "relevant_history": memory_vars.get("relevant_history", "")
        }
        response=chain.invoke(chain_input)
        response=str(response).strip()
        if not response:
            response="That's an interesting question about internships and final year projects! Looking at it positively, both paths offer unique opportunities for growth. Internships provide valuable real-world experience and networking, while final year projects let you showcase your expertise and innovation. Let's explore what excites you most about each option!"
        #confidence calculation
        confidence=calculate_optimist_confidence(user_input,state.topic_type)

        new_state.update_agent_response("optimist",response,confidence)

        #save to memory
        store_key=f"optimist_{topic}"
        save_to_memory(store_key,user_input,response)

    except Exception as e:
        print(f"Optimist error: {str(e)}")
        traceback.print_exc()
        fallback_response=(
            "Looking at this optimistically, both internships and final year projects offer amazing opportunities! "
            "Would you like to explore the exciting potential of each path?"
        )
        new_state.update_agent_response("optimist",fallback_response, 0.5)

    return new_state