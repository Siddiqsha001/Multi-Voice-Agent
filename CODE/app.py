from langgraph.graph import StateGraph, END
from planner import planner_node
from optimistic import optimistic_node
from realistic import realistic_node
from state import AgentState, initialize_state, ConversationEntry
from typing import Dict, Any, Union
import traceback
import uuid

workflow=StateGraph(AgentState)

workflow.add_node("planner",planner_node)
workflow.add_node("relaistic",realistic_node)
workflow.add_node("optimistic",optimistic_node)

def system_intro_node(state:AgentState)->AgentState:
    new_state=AgentState(**state.dict())
    new_state.active_agent="system"
    new_state.final_response="Hi! I'm here with my colleagues to discuss any topic you'd like. What would you like us to explore together?"
    return new_state
    
workflow.add_node("system_intro",system_intro_node)
workflow.set_entry_point("system_intro")

#handoff logic
def select_next(state:AgentState)->str:
    try:
        if not state.user_input.strip():
            return END
        if state.active_agent=="system":
            if not state.topic_type:
                if "internship" in state.user_input.lower() or "job" in state.user_input.lower():
                    state.topic_type="career"
                elif "study" in state.user_input.lower() or "course" in state.user_input.lower():
                    state.topic_type="education"
                elif any(tech in state.user_input.lower() for tech in ["programming", "software", "technology", "code"]):
                    state.topic_type="technical"
                else:
                    state.topic_type="career"  
            
            if not state.optimist_response:
                return "optimist"
            if not state.realist_response:
                return "realist"
            if not state.planner_response:
                return "planner"
            
        if state.topic_type=="career":
            if not state.optimist_response:
                return "optimist"
            elif not state.realist_response:
                return "realist"
            elif not state.planner_response:
                return "planner"
        elif state.topic_type=="education":
            if not state.realist_response:
                return "realist"
            elif not state.optimist_response:
                return "optimist"
            elif not state.planner_response:
                return "planner"
        elif state.topic_type=="technical":
            if not state.planner_response:
                return "planner"
            elif not state.realist_response:
                return "realist"
            elif not state.optimist_response:
                return "optimist"

        return END
    except Exception as e:
        print(f"Error in select_next: {str(e)}")
        traceback.print_exc()
        return END
    
for node in ["system_intro","planner","realist","optimistic"]:
    workflow.add_transition(node,select_next,{
        "system_intro":"system_intro",
            "planner":"planner",
            "realist":"realist",
            "optimist":"optimist",
            END:END
    })

app=workflow.compile()

def create_initial_state(user_input:str,is_voice:bool=False)->AgentState:
    return AgentState(
        user_input=user_input,
        is_voice_input=is_voice,
        session_id=str(uuid.uuid4()),
        active_agent=None,
        final_response=None,
        topic_type=None,
        user_emotion=None,
        user_intent=None,
        realist_response=None,
        optimist_response=None,
        planner_response=None,
        realist_confidence=0.0,
        optimist_confidence=0.0,
        planner_confidence=0.0,
        web_results=None,
        memory_context=None
    )

def extract_final_state(result:Union[Dict[str, Any],AgentState])->AgentState:
    try:
        if isinstance(result,AgentState):
            return result
        if isinstance(result,dict):
            if len(result)==1:
                inner=next(iter(result.values()))
                return inner if isinstance(inner,AgentState) else AgentState(**inner)
            return AgentState(**result)
        raise ValueError(f"Unexpected result structure type: {type(result)}")
    except Exception as e:
        print(f"Error extracting final state: {str(e)}")
        raise

def run_conversation(user_input:str, is_voice:bool=False)->AgentState:
    try:
        if not user_input or not isinstance(user_input,str):
            return AgentState(
                user_input=str(user_input),
                final_response="I didn't receive valid input. Could you try again?",
                active_agent="system",
                is_voice_input=is_voice
            )
        initial_state=create_initial_state(user_input, is_voice)

        if any(word in user_input.lower() for word in ["internship", "project", "job", "career", "work"]):
            initial_state.topic_type="career"

        try:
            result=app.invoke(initial_state)
            result_state=extract_final_state(result)
        except Exception as e:
            print(f"Workflow error: {str(e)}")
            print(traceback.format_exc())
            return AgentState(
                user_input=user_input,
                is_voice_input=is_voice,
                final_response="I'm sorry, I had trouble processing that. For questions about internships or projects, I can help you weigh the pros and cons. Would you like to try again?",
                active_agent="system"
            )

        if not result_state.final_response:
            lines=[]
            
            if not result_state.conversation_history:
                lines.append('**System**:"Hi! I\'m here with my colleagues to discuss any topic you\'d like. What would you like us to explore together?"')
            #to bold the sender
            lines.append(f'**User**: "{result_state.user_input.strip()}"')

            agent_configs = {
                "optimist":("Optimist Agent","in a hopeful tone"),
                "realist":("Realist Agent","in a factual tone"),
                "planner":("Planner Agent","in a strategic tone")
            }

            for agent_type, response in result_state.response_buffer.items():
                if response and agent_type in agent_configs:
                    agent_icon,tone=agent_configs[agent_type]
                    lines.append(f'**{agent_icon}** *({tone})*: "{response.strip()}"')

            if result_state.topic_type=="career":
                lines.append('**System**:"Would you like to explore more specific aspects of either the internship or project path?"')
            else:
                lines.append('**System**:"Would you like to go deeper into any of those points?"')
            result_state.final_response="\n\n".join(lines)
            result_state.active_agent="system"
        return result_state

    except Exception as e:
        print(f"Error in run_conversation: {str(e)}")
        print(traceback.format_exc())
        return AgentState(
            user_input=str(user_input),
            is_voice_input=is_voice,
            final_response="I encountered an issue, but I'm here to help. Could you try rephrasing your question about internships or projects?",
            active_agent="system"
        )

