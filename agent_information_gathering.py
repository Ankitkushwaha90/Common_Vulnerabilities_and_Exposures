import os
import streamlit as st
from typing import List, Dict, Any
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import PyPDF2
import docx
from langchain_community.document_loaders import WebBaseLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MODEL_PATH = "./deepseek-llm-7b-chat-Q6_K.gguf"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TOP_P = 0.95
REPEAT_PENALTY = 1.1
N_CTX = 2048

class AIChatApp:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.chat_history = []
        
    def _initialize_llm(self):
        """Initialize the LLaMA model with GGUF format."""
        # Check if model file exists and is not malicious
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            st.error(f"Model file not found at: {MODEL_PATH}. Please ensure the file exists.")
            return None
        
        if os.path.basename(MODEL_PATH) == "malicious.gguf":
            logger.error("Attempted to load malicious GGUF file. Aborting.")
            st.error("Error: Cannot load 'malicious.gguf'. Please use a legitimate model file.")
            return None
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        try:
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                repeat_penalty=REPEAT_PENALTY,
                n_ctx=N_CTX,
                callback_manager=callback_manager,
                verbose=True,
                n_gpu_layers=0  # Use CPU to avoid GPU-related issues
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
            return None
        return llm
    
    def _extract_text_from_file(self, file):
        """Extract text from uploaded file based on its type."""
        file_extension = os.path.splitext(file.name)[1].lower()
        text = ""
        
        try:
            if file_extension == '.pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif file_extension == '.txt':
                text = file.read().decode('utf-8')
            elif file_extension == '.docx':
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                text = "Unsupported file format. Please upload PDF, TXT, or DOCX files."
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            text = f"Error processing file: {str(e)}"
            
        return text
    
    def _extract_text_from_url(self, url: str) -> str:
        """Extract text from a website URL using WebBaseLoader."""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            logger.error(f"Error fetching website content: {str(e)}")
            return f"Error fetching website content: {str(e)}"
    
    def stream_response(self, user_input: str, context: str = "") -> str:
        """Stream the response from the LLaMA model."""
        if not self.llm:
            return "Error: Model not loaded. Please check the model file and try again."
        
        template = """
        You are a helpful AI assistant specialized in information gathering. Provide a detailed and helpful response to the user's query.
        If relevant, use the provided context (from documents or websites) to inform your response. Filter and summarize key information as needed.
        
        Context (if any):
        {context}
        
        Chat History:
        {chat_history}
        
        User: {user_input}
        Assistant: """
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "user_input"],
            template=template
        )
        
        # Create a streaming callback handler
        from langchain.callbacks.base import BaseCallbackHandler
        
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text = initial_text
            
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                self.container.markdown(self.text + "▌")
        
        # Create a placeholder for the streaming output
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Configure the LLM with streaming
        self.llm.callbacks = [stream_handler]
        
        # Run the chain
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        try:
            response = llm_chain.run(
                context=context,
                chat_history="\n".join(self.chat_history[-4:]),
                user_input=user_input
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = f"Error generating response: {str(e)}"
        
        # Update the final response without the cursor
        response_placeholder.markdown(response)
        
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"Assistant: {response}")
        
        return response

def main():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Assistant with Information Gathering")
    st.markdown("Chat with your personal AI assistant powered by DeepSeek LLM. Upload documents or provide website links for context.")
    
    # Initialize session state
    if 'chat_app' not in st.session_state:
        st.session_state.chat_app = AIChatApp()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or DOCX)", 
                                   type=['pdf', 'txt', 'docx'])
    
    # URL input for website
    website_url = st.text_input("Enter a website URL for information gathering")
    
    # Process uploaded file and URL
    context = ""
    file_content = ""
    if uploaded_file is not None:
        file_content = st.session_state.chat_app._extract_text_from_file(uploaded_file)
        context += f"Document Content:\n{file_content}\n\n"
        st.success("File uploaded successfully!")
        with st.expander("View uploaded document content"):
            st.text_area("Document Content", file_content, height=200)
    
    web_content = ""
    if website_url:
        web_content = st.session_state.chat_app._extract_text_from_url(website_url)
        context += f"Website Content from {website_url}:\n{web_content}\n\n"
        st.success("Website content fetched successfully!")
        with st.expander("View website content"):
            st.text_area("Website Content", web_content, height=200)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response with streaming
        with st.chat_message("assistant"):
            response = st.session_state.chat_app.stream_response(prompt, context)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
