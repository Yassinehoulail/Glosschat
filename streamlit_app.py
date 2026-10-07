"""Interface Streamlit simple pour le chatbot KPI."""

import uuid

import streamlit as st

from rag import ask


st.set_page_config(page_title="GlossChat", page_icon="💬", layout="centered")

st.title("💬 GlossChat")
st.caption("Posez vos questions sur les KPI présents dans les glossaires indexés.")

if st.sidebar.button("Nouvelle conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Quel KPI souhaitez-vous rechercher ?",
        }
    ]
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Quel KPI souhaitez-vous rechercher ?",
        }
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Exemple : Quelle est la définition de DMC ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les glossaires..."):
            try:
                answer = ask(question, thread_id=st.session_state.thread_id)
            except Exception as error:
                answer = f"Une erreur est survenue : {error}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
