#manages current convo flow
#combines buffer and vector memory
import os
from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from memory_store import store_memory,get_memories,clear_memories
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)

try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Error initializing LLM: {str(e)}")
    raise

class VectorEnhancedMemory(ConversationBufferMemory):
    #memory class that combines buffer memory with vector store
    def __init__(self, agent_type: str):
        super().__init__(
            memory_key="history",
            input_key="user_input",
            output_key="output",
            return_messages=True
        )
        self._agent_type=agent_type

    @property
    def agent_type(self)->str:
        return self._agent_type

    def format_memory(self, messages)->str:
        formatted=[]
        for msg in messages:
            if hasattr(msg, 'content'):
                formatted.append(f"{msg.type.capitalize()}: {msg.content}")
        return "\n".join(formatted) if formatted else "No history available."

    def get_relevant_history(self,query:str,top_k:int=3)->str:
        try:
            memories=get_memories(query=query,agent_type=self.agent_type,top_k=top_k)
            return "\n\n".join([
                f"{m['text']} (relevance: {m['relevance_score']:.2f})"
                for m in memories
            ])
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return "No relevant history available."

    def load_memory_variables(self,inputs: Dict[str, Any])->Dict[str,Any]:
        try:
            buffer_memory=super().load_memory_variables(inputs)
            query=inputs.get(self.input_key, "")
            relevant_memories=get_memories(query=query,agent_type=self.agent_type,top_k=3)
            relevant_history="\n".join([
                f"Memory {i+1} (Score: {mem['relevance_score']:.2f}):\n{mem['text']}"
                for i, mem in enumerate(relevant_memories)
            ]) if relevant_memories else "No relevant memories found."
            return {
                "history":self.format_memory(buffer_memory.get("history", [])),
                "relevant_history":relevant_history,
                self.input_key:query
            }
        except Exception as e:
            print(f"Error loading memory variables: {str(e)}")
            return {
                "history": "No history available.",
                "relevant_history": "No relevant memories found.",
                self.input_key: inputs.get(self.input_key, "")
            }

    def save_context(self,inputs:Dict[str, Any],outputs:Dict[str, str])->None:
        super().save_context(inputs, outputs)
        try:
            store_memory(
                agent_type=self.agent_type,
                user_input=inputs.get(self.input_key, ""),
                agent_response=outputs.get(self.output_key, "")
            )
        except Exception as e:
            print(f"Warning: Failed to store memory in vector store: {str(e)}")

agent_memories={}

try:
    agent_memories={
        "optimist": VectorEnhancedMemory(agent_type="optimist"),
        "realist": VectorEnhancedMemory(agent_type="realist"),
        "planner": VectorEnhancedMemory(agent_type="planner")
    }
    print("Successfully initialized agent memories")
except Exception as e:
    print(f"Error initializing memories: {str(e)}")
    #fallback to regular memory if vector enhanced fails
    for agent_type in ["optimist", "realist", "planner"]:
        agent_memories[agent_type]=ConversationBufferMemory(
            memory_key="history",
            input_key="user_input",
            output_key="output",
            return_messages=True
        )

def save_to_memory(agent_type:str,user_input:str,response:str)->None:
    try:
        if agent_type in agent_memories:
            memory=agent_memories[agent_type]
            memory.save_context(
                {"user_input":user_input},
                {"output":response}
            )
    except Exception as e:
        print("")

def clear_all_memories()->None:
    try:
        for memory in agent_memories.values():
            memory.clear()
        clear_memories()
    except Exception as e:
        print(f"Error clearing memories: {str(e)}")
