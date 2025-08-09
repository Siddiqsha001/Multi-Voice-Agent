import os
import requests
from dotenv import load_dotenv
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
def search_web(query):
    url="https://google.serper.dev/search"
    headers={
        "X-API-KEY":SERPER_API_KEY,
        "Content-Type":"application/json"
    }
    payload={
        "q":query
    }
    try:
        response=requests.post(url,headers=headers,json=payload)
        response.raise_for_status()
        data=response.json()
        if "organic" not in data:
            return "No web results found."
        top_results=data["organic"][:3]
        formatted_results=[]
        for result in top_results:
            title=result.get("title", "")
            snippet=result.get("snippet", "")
            link=result.get("link", "")
            formatted_results.append(f" {title}\n{snippet}\n {link}")
        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {e}"
