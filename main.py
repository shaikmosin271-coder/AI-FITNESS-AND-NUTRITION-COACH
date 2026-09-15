import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.set_page_config(page_title="Workout Split & Diet Q&A", page_icon="🌱")
st.title("Workout Split & Diet Q&A 🌱")

# Cache QA chain so models and FAISS index don't re-load on every button click
@st.cache_resource
def load_qa_chain():
    return get_qa_chain()

# Button to create or update the vector database
btn = st.sidebar.button("Create Knowledgebase")
if btn:
    with st.spinner("Building vector database..."):
        create_vector_db()
        st.cache_resource.clear()  # Clear cache to reload updated FAISS index
        st.sidebar.success("Knowledgebase created successfully!")

# Input field for user question
question = st.text_input("Question: ", placeholder="e.g., What are the macros for Chicken Breast?")

if question:
    with st.spinner("Searching fitness database..."):
        try:
            chain = load_qa_chain()
            # FIX: Use .invoke() instead of direct function call
            response = chain.invoke({"query": question})

            st.header("Answer")
            st.write(response["result"])
        except Exception as e:
            st.error(f"Error fetching answer: {str(e)}")
            st.info("If you haven't built the database yet, click 'Create Knowledgebase' in the sidebar.")