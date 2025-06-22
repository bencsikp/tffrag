import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Custom CSS
st.markdown("""
    <style>
        .response-area {
            font-size: 1.4rem;
            background-color: #f9f9f9;
            padding: 1em;
            border-radius: 8px;
            margin-top: 1em;
            white-space: pre-wrap;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Top Female Founders Summer School friendly information bot")

# Available OpenAI models
AVAILABLE_MODELS = {
    "gpt-4.1": "GPT-4.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-3.5-turbo": "GPT-3.5 Turbo"
}

# Vector store path
VECTOR_DB_PATH = "vector_store"

def test_openai_connection(api_key, model_name):
    """Test OpenAI API connection and model availability"""
    if not api_key:
        return False, "OpenAI API key not found in environment variables"
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "OpenAI API connection successful"
    
    except openai.AuthenticationError:
        return False, "Invalid OpenAI API key"
    except openai.NotFoundError:
        return False, f"Model '{model_name}' not found or not accessible"
    except openai.RateLimitError:
        return False, "Rate limit exceeded. Please try again later"
    except openai.APIConnectionError:
        return False, "Failed to connect to OpenAI API. Check your internet connection"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

@st.cache_resource
def load_vector_store():
    """Load the FAISS vector store if it exists"""
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                VECTOR_DB_PATH,
                OpenAIEmbeddings(openai_api_key=openai_api_key),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None
    else:
        return None

def create_qa_chain(model_name, vector_store):
    """Create a QA chain with the specified model"""
    try:
        llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=openai_api_key,
            model=model_name
        )
        
        # Define a more specific prompt template
        from langchain.prompts import PromptTemplate
        prompt_template = """You are a helpful assistant providing information about the onsite week.
        Answer the question based ONLY on the provided context.
        If the answer is not found in the context, clearly state that you don't know or that the information is not available.
        Be precise and directly answer the question without additional conversational text.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}), # Increased k to retrieve more documents
            return_source_documents=True, # Added to easily see what documents were retrieved
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

selected_model = "gpt-4.1"

# Main application logic
def main():
    if not openai_api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    vector_store = load_vector_store()
    
    if vector_store is None:
        st.warning("⚠️ No vector store found. Please run `python ingest.py` to index your documents first.")
        st.info("Make sure you have documents in the 'docs' folder before running the ingestion script.")
        return
    
    with st.spinner("Initializing... Testing OpenAI connection..."):
        is_connected, connection_message = test_openai_connection(openai_api_key, selected_model)
    
    if not is_connected:
        st.error(f"❌ {connection_message}")
        st.info("Please check your API key and internet connection, then refresh the page.")
        return
    
    st.success(f"✅ Connected to OpenAI API using {AVAILABLE_MODELS[selected_model]}")
    
    qa_chain = create_qa_chain(selected_model, vector_store)
    
    if qa_chain is None:
        st.error("Failed to initialize the QA system. Please check your configuration.")
        return
    
    st.subheader("Ask your question")
    user_question = st.text_input(
        "Enter your question about the documents:",
        placeholder="e.g., What are the key topics covered in the documents?"
    )
    
    if user_question:
        with st.spinner("Searching for relevant information..."):
            try:
                response = qa_chain.invoke({"query": user_question})
                answer = response.get("result", "No answer found")
                st.markdown(f'<div class="response-area">{answer}</div>', unsafe_allow_html=True)
                
                #with st.expander("Source Information"):
                #    source_documents = response.get("source_documents", [])
                #    if source_documents:
                #        for i, doc in enumerate(source_documents, 1):
                #            st.write(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                #            st.write(f"**Content preview:** {doc.page_content[:500]}...") # Increased preview length
                #            st.write("---")
                #    else:
                #        st.write("No source documents found for this query.")
                
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                st.info("Please try rephrasing your question or check the system configuration.")

if __name__ == "__main__":
    main()