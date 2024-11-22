import os
import streamlit as st
from streamlit_chat import message
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Set up Groq API and model
os.environ['GROQ_API_KEY'] = "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"  # Replace with your actual API key
model = 'llama3-8b-8192'
groq_chat = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name=model)
system_prompt = 'You are a friendly conversational chatbot that helps with Indian law-related queries. For general conversations, you can chat normally.'
conversational_memory_length = 5

# Initialize conversational memory
memory = ConversationBufferWindowMemory(
    k=conversational_memory_length,
    memory_key="chat_history",
    return_messages=True
)

# Load the vectorstore (knowledge base of laws)
with open('vector_store.pkl', 'rb') as f:
    vectorstore = pickle.load(f)

# Set up OpenAI embeddings for vector search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a lightweight model

def is_law_related(user_question):
    """Determine if the user's question is related to law."""
    law_keywords = ['law', 'court', 'act', 'regulation', 'legal', 'rights', 'rules', 'justice']
    return any(keyword in user_question.lower() for keyword in law_keywords)

def search_law_database(user_question):
    """Search the law database for relevant information."""
    user_query_embedding = embedding_model.encode([user_question])[0]
    
    # Load precomputed embeddings
    with open('vectorstore_embeddings.npy', 'rb') as f:
        stored_embeddings = np.load(f)
    
    # Perform cosine similarity calculation
    similarities = cosine_similarity([user_query_embedding], stored_embeddings)
    
    # Get the index of the top 3 most similar documents
    top_indices = similarities[0].argsort()[-3:][::-1]
    
    # Retrieve corresponding document content
    with open('vectorstore_content.pkl', 'rb') as f:
        stored_documents = pickle.load(f)
    
    return "\n".join([stored_documents[i] for i in top_indices])

# Streamlit setup
st.title("DOJ ChatBot ⚖️")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'generated' not in st.session_state:
    st.session_state.generated = ["Hello! Ask me anything about Indian law"]

if 'past' not in st.session_state:
    st.session_state.past = ["Hey! 👋"]

def conversation_chat(query):
    """Generate chatbot response."""
    # Check if the question is law-related
    if is_law_related(query):
        law_info = search_law_database(query)
    else:
        law_info = ""

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
            HumanMessagePromptTemplate.from_template(f"{law_info}"), 
        ]
    )   
    
    # Create conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory
    )
    
    # Get response
    response = conversation.predict(human_input=query)
    
    # Append to chat history
    st.session_state.chat_history.append((query, response))
    return response

# Chat UI with streamlit-chat
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Ask about Indian law:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    # Display chat history with streamlit_chat
    if st.session_state.generated:
        with reply_container:
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state.generated[i], key=str(i), avatar_style="fun-emoji")

# Display the chat history
display_chat_history()
