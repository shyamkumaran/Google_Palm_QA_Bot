import streamlit as st
from KB_Helper import create_vectordb, quest_ans_chain

st.title("Hotel Montreal Chatbot")

btn = st.button("Create KB")

if btn:
    create_vectordb()

question = st.text_input("Question: ")

if question:
    chain = quest_ans_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
