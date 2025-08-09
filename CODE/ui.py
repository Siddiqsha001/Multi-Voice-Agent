import streamlit as st
from tts_stt import listen_and_transcribe,speak
from app import run_conversation
from state import initialize_state,AgentState
import time
st.set_page_config(page_title="Multi-Agent Voice System",layout="centered")
st.title("Talk with the Agents")

if "agent_state" not in st.session_state:
    st.session_state.agent_state=initialize_state()

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

#for history display
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)

def run_and_speak(final_state: AgentState):
    agents=[
        ("optimist","Optimist Agent","optimist"),
        ("realist","Realist Agent","realist"),
        ("planner","Planner Agent","planner")
    ]
    for key,label,agent_key in agents:
        response=getattr(final_state, f"{key}_response")
        if response and response.strip():
            with st.chat_message(agent_key):
                if key=="realist" and "SOURCES:" in response:
                    main_content, sources=response.split("SOURCES:", 1)
                    st.success(f"**{label}**: {main_content.strip()}")
                    speak(f"{main_content.strip()} Please refer to the sources below for more information.", agent=agent_key)
                    st.info("**Sources:**\n" + sources.strip())
                    st.session_state.chat_history.append((agent_key, f"**{label}**: {main_content.strip()}\n\n**Sources:**\n{sources.strip()}"))
                else:
                    st.success(f"**{label}**: {response}")
                    speak(response,agent=agent_key)
                    st.session_state.chat_history.append((agent_key, f"**{label}**: {response}"))
                time.sleep(len(response.split()) * 0.3) 

if st.button("Start Talking"):
    with st.chat_message("user"):
        st.info("Listening... Speak now...")
    user_input=listen_and_transcribe()

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        try:
            final_state=run_conversation(user_input)

            if isinstance(final_state, AgentState):
                run_and_speak(final_state)
                st.session_state.agent_state=final_state
            else:
                raise ValueError("Invalid AgentState returned")

        except Exception as e:
            error_msg=f"An error occurred: {str(e)}"
            st.session_state.chat_history.append(("system", error_msg))
            with st.chat_message("system"):
                st.error(error_msg)
    else:
        with st.chat_message("system"):
            st.error("Could not detect any voice input. Please try again.")
