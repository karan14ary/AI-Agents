##Basic AI Agent with Memory using Streamlit webUI

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

#loadAI Model for Ollama
llm = OllamaLLM(model="mistral")
#initialize message history/memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

    #define AI assistant prompt
prompt=PromptTemplate(
    input_variables=["chat_history", "question"],
    template="previous conversation: {chat_history}\n\n current question: {question}\n\n AI: "
)
#Function to run AI chat with memory
def run_chain(question):
    #Retriev chat history as from memory
    chat_history_text="\n".join([f"{message.type}: {message.content}" for message in st.session_state.chat_history.messages])
    #Run the AI response geneation
    response=llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    #store the question and response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

#streamlit app interface
st.title("AI Assistant with Memory")
st.write("Ask me anything! Type 'exit' to stop the conversation.")
user_input = st.text_input("Your Question:")
if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye!")
    else:
        ai_response = run_chain(user_input)
        st.write(f"Your Question: {user_input}")
        st.write(f"AI Response: {ai_response}")         
#show full conversation history
st.subheader("Conversation History")
for message in st.session_state.chat_history.messages:
    st.write(f"{message.type.capitalize()}: {message.content}")
    

##Basic AI Agent with Memory

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM

# #loadAI Model for Ollama
# llm = OllamaLLM(model="mistral")

# #initialize message history/memory
# chat_history = ChatMessageHistory()
# #define AI assistant prompt
# prompt=PromptTemplate(
#     input_variables=["chat_history", "quesation"],
#     template="previous conversation: {chat_history}\n\n current question: {question}\n\n AI: "
# )
# #Function to run AI chat with memory
# def run__chain(question):
#     #Retriev chat history as from memory
#     chat_history_text="\n".join([f"{message.type}: {message.content}" for message in chat_history.messages])
#     #Run the AI response geneation
#     response=llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
#     #store the question and response in memory
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response)
#     return response

# #interactive CLI chatBot
# print("\n AI Assistant with Memory, Ask me anything! \n")
# print("Type 'exit' to stop the conversation. \n")
# while True:
#     user_input = input("Your :")
#     if user_input.lower() == "exit":
#         print("Goodbye!")
#         break
#     ai_response = run__chain(user_input)
#     print(f"\n AI Response: {ai_response}")

    ##Basic AI Agent without memory

# from langchain_ollama import OllamaLLM

# #loadAI Model for Ollama
# llm = OllamaLLM(model="mistral")
# print("\n Welcome too your AI Assistant, Ask me anything! \n")
# while True:
#     question = input("Your Questaions (or type 'exit' to stop): ")
#     if question.lower() == "exit":
#         print("Goodbye!")
#         break
#     response = llm.invoke(question)
#     print("\n AI Response: ", response)