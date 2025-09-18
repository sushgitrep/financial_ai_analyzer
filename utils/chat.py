"""
Simple PDF Chatbot - Core Logic
Handles PDF processing, vector storage, and chat with memory
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Vector database imports
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
# Wrap it in a Document object
from langchain.schema import Document

# Configuration
import chatbot_config
import streamlit as st
from loguru import logger

class SimplePDFChatbot:
    """Simple PDF chatbot with configurable vector database and memory"""

    def __init__(self):
        
        st.session_state.chatbot = None
        st.session_state.messages = []
    
        """Initialize the chatbot"""
        # Validate configuration
        errors = chatbot_config.validate_config()
        if errors:
            for error in errors:
                print(error)
            raise ValueError("Configuration validation failed")

        logger.info("Initialize components")
        # Initialize components
        self.llm = ChatOpenAI(
            api_key=chatbot_config.OPENAI_API_KEY,
            model=chatbot_config.OPENAI_MODEL,
            temperature=chatbot_config.TEMPERATURE,
            max_tokens=chatbot_config.MAX_TOKENS
        )

        logger.info("OpenAIEmbeddings")
        self.embeddings = OpenAIEmbeddings(api_key=chatbot_config.OPENAI_API_KEY)

        logger.info("RecursiveCharacterTextSplitter")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chatbot_config.CHUNK_SIZE,
            chunk_overlap=chatbot_config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Chat components (initialized after PDF processing)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

        # Simple chat history storage
        self.chat_history: List[BaseMessage] = []
            
        if "filepath" in st.session_state:
            logger.info("filepath in st.session_state:")
            print(f"📄 Loading default PDF: {st.session_state.get("filepath")}")
            self.process_pdf(st.session_state.get("filepath"))

    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a PDF file and set up the RAG chain

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if successful, False otherwise
        """
        logger.info("process_pdf in simplepdfchatbot")
        try:
            print(f"📄 Processing PDF: {pdf_path}")

            # # Load PDF
            # loader = PyPDFLoader(pdf_path)
            # documents = loader.load()
            # print(f"✅ Loaded {len(documents)} pages")
            logger.info("reading raw text")
            
            documents = [Document(page_content=st.session_state.raw_text)]
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"✂️ Created {len(chunks)} chunks")

            # Create vector store
            if chatbot_config.VECTOR_DB.lower() == "chroma":
                # Use ChromaDB
                self.vectorstore = Chroma.from_documents(
                    chunks, 
                    self.embeddings
                )
                print("✅ Created ChromaDB vector store")
            else:
                # Use FAISS (default)
                self.vectorstore = FAISS.from_documents(
                    chunks, 
                    self.embeddings
                )
                print("✅ Created FAISS vector store")

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": chatbot_config.SIMILARITY_SEARCH_K}
            )

            # Create RAG chain
            self._setup_rag_chain()

            print("🎉 PDF processed successfully! Ready to chat.")
            st.session_state.pdf_processed  = True 
            return True

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return False

    def _setup_rag_chain(self):
        """Set up the RAG chain with chat history"""
        logger.info("set up rag chian")
        # Create prompt template with memory
        system_prompt = (
            "You are a helpful AI assistant that answers questions about the provided document. "
            "Use the following pieces of context to answer the user's question. "
            "If you don't know the answer based on the context, just say you don't know. "
            "Keep your answers concise and relevant.\n\n"
            "Context: {context}"
        )

        if chatbot_config.MAX_CHAT_HISTORY > 0:
            # Include chat history in prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        else:
            # No chat history
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        self.rag_chain = create_retrieval_chain(self.retriever, document_chain)

        print("✅ RAG chain created with chat history support")

    def _get_recent_history(self) -> List[BaseMessage]:
        """Get recent chat history based on chatbot_config"""
        if chatbot_config.MAX_CHAT_HISTORY <= 0:
            return []

        # Get last N exchanges (N*2 messages since each exchange = human + ai)
        max_messages = chatbot_config.MAX_CHAT_HISTORY * 2
        return self.chat_history[-max_messages:] if len(self.chat_history) > max_messages else self.chat_history

    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get response

        Args:
            message: User message

        Returns:
            Bot response
        """
        if not self.rag_chain:
            return "❌ Please upload and process a PDF file first."

        try:
            # Prepare input
            chain_input = {
                "input": message
            }

            # Add chat history if enabled
            if chatbot_config.MAX_CHAT_HISTORY > 0:
                chain_input["chat_history"] = self._get_recent_history()

            # Get response from RAG chain
            response = self.rag_chain.invoke(chain_input)
            answer = response["answer"]

            # Update chat history
            if chatbot_config.MAX_CHAT_HISTORY > 0:
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=answer))

            return answer

        except Exception as e:
            error_msg = f"❌ Error generating response: {e}"
            print(error_msg)
            return error_msg

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        print("✅ Chat history cleared")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get chat history in a simple format

        Returns:
            List of message dictionaries
        """
        messages = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages

def save_uploaded_file(uploaded_file) -> str:

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def main():

    # Initialize chatbot
    try:
        bot = SimplePDFChatbot()
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return

# if __name__ == "__main__":
#     main()
