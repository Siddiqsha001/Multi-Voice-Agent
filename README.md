# Hi there, I am Siddiqsha

The given task is about a **Multi Agent voice conversation** system.

*Built a multi agent voice assistant with deepgram and pinecone integration. Designed modular agents (optimist, realist and planner) with shared vector memory space and real time web tools, orchestrated through streamlit UI.*

- This enables the user to interact or discuss with the agents in own pace.

**Implementation:**

- I tested it with 2 scenarios.
- One is **technical + real time related** while the other one is **technical + plan related**.

**Key takeaways:**

- To deal with voice input and output(Voice Integration).
- Proper flow implementation amoung the agents.
- Implemented a shared space which is accessible by all the agents/nodes.
- Vector based knowledge + real time web search.
- Usage of fallback strategies for debugging.

**Challenges:**

- Voice input accuracy : used deepgram for converting speech to text.
- Decentralized agent flow : implemented proper flow using LangGraph.
