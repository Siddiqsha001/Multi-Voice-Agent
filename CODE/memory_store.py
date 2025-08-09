#for storing conversations in Pinecone
#semantic memory store, retrieving similar past conversations- long term memory
import os
import json
import logging
from uuid import uuid4
from typing import List,Dict,Any,Optional
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
import numpy as np
import google.generativeai as genai
from google.generativeai import GenerativeModel


load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

INDEX_NAME="voiceagent"
EMBEDDING_DIM=1024
NAMESPACE="default"

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

try:
    pc=Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone client initialized.")

    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index=pc.Index(INDEX_NAME)
    logger.info(f"Connected to index: {INDEX_NAME}")

except Exception as e:
    logger.error(f"Pinecone initialization failed: {str(e)}")
    raise
#text to vector embedding function
def encode_text_to_embedding(text: str) -> List[float]:
    try:
        model = GenerativeModel('models/embedding-001')
        result = model.embed_content(text)
        return result.embedding
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return [0.0] * EMBEDDING_DIM

def store_memory(user_input: str,agent_response: str,agent_type: str,
                emotion: Optional[str]=None,context: Optional[Dict[str, Any]]=None)->None:
    try:
        memory_id=str(uuid4())
        full_text=f"User: {user_input}\nAgent: {agent_response}"
        embedding=encode_text_to_embedding(full_text)

        metadata={
            "user_input": user_input,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "text": full_text,
            "timestamp": str(np.datetime64('now'))
        }

        if emotion:
            metadata["emotion"]=emotion
        if context:
            metadata["context"]=json.dumps(context)

        vector={
            "id":memory_id,
            "values":embedding,
            "metadata":metadata
        }

        index.upsert(vectors=[vector],namespace=NAMESPACE)
        logger.info(f"Stored memory ID {memory_id} in namespace: {NAMESPACE}")

    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise
#to retrieeve similar past conversations
def get_memories(query: str, agent_type: Optional[str]=None,top_k:int=4)->List[Dict[str, Any]]:
    try:
        query_embedding=encode_text_to_embedding(query)

        filter_dict={"agent_type": {"$eq": agent_type}} if agent_type else None

        results=index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
            filter=filter_dict
        )

        matches = results.get("matches", [])
        logger.info(f"Retrieved {len(matches)} matches for query: {query}")

        return [{
            "text":match["metadata"].get("text", ""),
            "agent_type":match["metadata"].get("agent_type", ""),
            "relevance_score":match.get("score", 0.0)
        } for match in matches]

    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}")
        raise

def clear_memories()->None:
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        logger.info(f"Cleared all vectors in namespace: {NAMESPACE}")
    except Exception as e:
        logger.error(f"Error clearing memories: {str(e)}")
        raise
