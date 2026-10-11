
import streamlit as st

from app.rag_chat import (
    build_components,
    answer_question,
)

st.set_page_config(
    page_title="Arogyini | Career Guidance",
    page_icon="🌷",
    layout="centered",
)

st.title("🌷 Arogyini")
st.subheader("Your career and livelihood companion")
st.write(
    "Explore career paths, skills, self-employment, "
    "training, and opportunities that fit your goals."
)

# Initialize the RAG components only once per session.
@st.cache_resource
def load_arogyini():
    return build_components()


try:
    vectorstore, llm, prompt = load_arogyini()
except Exception as error:
    st.error(
        "Arogyini could not start. Please check your API key, "
        "Chroma database, and terminal error."
    )
    st.exception(error)
    st.stop()

# Keep conversation history during the session.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! 🌷 I'm Arogyini. Tell me a little about "
                "your goals, education, interests, or the kind of "
                "livelihood you'd like to explore. You can start "
                "wherever you are."
            ),
        }
    ]

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new message.
question = st.chat_input("Ask Arogyini anything about your future...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Arogyini is thinking..."):
            try:
                answer = answer_question(
                    question,
                    vectorstore,
                    llm,
                    prompt,
                )
            except Exception:
                answer = (
                    "I'm sorry, I couldn't process that message. "
                    "Please try again."
                )

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# Sidebar controls.
with st.sidebar:
    st.header("About Arogyini")
    st.write(
        "A supportive guide for exploring careers, learning, "
        "self-employment, and livelihood options."
    )

    if st.button("Start a new conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome back! 🌷 What would you like "
                    "help exploring today?"
                ),
            }
        ]
        st.rerun()