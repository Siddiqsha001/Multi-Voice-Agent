from typing import Optional,Dict,Any,List,Tuple,Union
from pydantic import BaseModel,Field,model_validator

class ConversationEntry(BaseModel):
    agent:str
    message:str
    confidence:float

class AgentState(BaseModel):
    user_input:str=""
    active_agent:Optional[str]=None
    final_response:Optional[str]=None
    is_voice_input:bool=False

    #confidence
    realist_response:Optional[str]=None
    optimist_response:Optional[str]=None
    planner_response:Optional[str]=None
    realist_confidence:float=0.0
    optimist_confidence:float=0.0
    planner_confidence:float=0.0

    user_emotion:Optional[str]=None
    user_intent:Optional[str]=None
    topic_type:Optional[str]=None 

    web_results:Optional[str]=None
    memory_context:Optional[str]=None
    response_buffer:Dict[str, str]=Field(default_factory=dict)
    conversation_history:List[ConversationEntry]=Field(default_factory=list)
    #metadata
    user_id:str="user_001"
    session_id:Optional[str]=None

    is_concurrent:bool=True

    @model_validator(mode='before')
    @classmethod
    def validate_state(cls, values):
        if isinstance(values, dict):
            if 'user_input' in values:
                values['user_input']=str(values['user_input']) if values['user_input'] is not None else ""
            
            for conf in ['realist_confidence', 'optimist_confidence', 'planner_confidence']:
                if conf in values:
                    try:
                        values[conf]=float(values[conf])
                    except (TypeError, ValueError):
                        values[conf]=0.0
            
            if 'conversation_history' in values:
                if not isinstance(values['conversation_history'],list):
                    values['conversation_history']=[]
                else:
                    history=[]
                    for entry in values['conversation_history']:
                        if isinstance(entry, dict):
                            entry['confidence']=float(entry.get('confidence', 0.0))
                            history.append(ConversationEntry(**entry))
                    values['conversation_history']=history

            if 'topic_type' in values:
                valid_topics={'career','education','technical','general'}
                if values['topic_type'] not in valid_topics:
                    values['topic_type']='general'
        return values

    def update_agent_response(self, agent_type: str, response: str, confidence: float):
        if not response:
            return
            
        response=str(response).strip()
        confidence=max(0.0, min(1.0, float(confidence)))
        
        if agent_type=="realist":
            self.realist_response=response
            self.realist_confidence=confidence
        elif agent_type=="optimist":
            self.optimist_response=response
            self.optimist_confidence=confidence
        elif agent_type=="planner":
            self.planner_response=response
            self.planner_confidence=confidence
            
        self.conversation_history.append(ConversationEntry(
            agent=agent_type,
            message=response,
            confidence=confidence
        ))

    def get_best_response(self)->Tuple[str, str]:
        confidences = {
            "realist": self.realist_confidence,
            "optimist": self.optimist_confidence,
            "planner": self.planner_confidence
        }
        responses = {
            "realist": self.realist_response or "",
            "optimist": self.optimist_response or "",
            "planner": self.planner_response or ""
        }
        valid_agents=[agent for agent, resp in responses.items() if resp]
        if not valid_agents:
            return "system", "I'm here with my colleagues to help you make the best decision. Could you tell us more about what you're considering?"
        best_agent=max(valid_agents, key=lambda a: confidences[a])
        return best_agent, responses[best_agent]

    @model_validator(mode='after')
    def update_response_buffer(self)->'AgentState':
        if self.optimist_response and 'optimist' not in self.response_buffer:
            self.response_buffer['optimist']=self.optimist_response
        if self.realist_response and 'realist' not in self.response_buffer:
            self.response_buffer['realist']=self.realist_response
        if self.planner_response and 'planner' not in self.response_buffer:
            self.response_buffer['planner']=self.planner_response
        return self

def initialize_state()->AgentState:
    return AgentState()
