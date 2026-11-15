
import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

# ----------------------------------
# 1. Load environment variables
# ----------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ----------------------------------
# 2. Streamlit page configuration
# ----------------------------------

st.set_page_config(
    page_title="Rajmohan AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Rajmohan AI Chatbot")

st.caption("Powered by Python, LangChain and Groq")


# ----------------------------------
# 3. Check API key
# ----------------------------------

if not API_KEY:
    st.error("Groq API key is missing. Please configure your .env file.")
    st.stop()


# ----------------------------------
# 4. Initialize Groq model
# ----------------------------------

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=API_KEY
)


# ----------------------------------
# 5. Create prompt template
# ----------------------------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer questions clearly and accurately.
        Use simple English.
        Keep answers short.

        Return your response in the following JSON format:

        {
            "answer": "Your answer here",
            "sources": []
        }

        Rules:
        1. Always return valid JSON.
        2. Put the answer inside the answer field.
        3. Add source URLs only if you actually used or can verify them.
        4. If no sources are available, return an empty array.
        5. Do not include Markdown code fences.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# ----------------------------------
# 6. Create LangChain chain
# ----------------------------------

chain = prompt_template | llm | JsonOutputParser()


# ----------------------------------
# 7. Initialize chat history
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# 8. Sidebar
# ----------------------------------

with st.sidebar:

    st.header("Settings")

    st.write(f"Model: {MODEL_NAME}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------
# 9. Display previous messages
# ----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------------
# 10. Get user input
# ----------------------------------

user_question = st.chat_input("Ask me anything and type hear...")


if user_question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(user_question)

    # Save user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Convert previous messages into LangChain format
    chat_history = []

    for message in st.session_state.messages[:-1]:

        if message["role"] == "user":
            chat_history.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            chat_history.append(
                AIMessage(content=message["content"])
            )

    # Keep recent conversation history
    chat_history = chat_history[-20:]

    # Generate AI response
    with st.chat_message("assistant"):

        try:

            response = chain.invoke({
                "chat_history": chat_history,
                "question": user_question
            })

            # Display answer
            st.markdown(response["answer"])

            # Display sources
            if response.get("sources"):
                with st.expander("Sources"):
                    for source in response["sources"]:
                        st.markdown(f"- {source}")

            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"]
            })

        except Exception as error:

            st.error(f"Something went wrong: {error}")
