
import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# MODEL DEFINITIONS
# =======================================================================

template =  """
            You are a helpful assistant that can scrape web pages, index their content, and answer questions based on the indexed data.
            You will be provided with a URL to scrape, and you will return the indexed content.
            You can also answer questions based on the indexed content.
            The content will be indexed using Ollama embeddings and stored in an in-memory vector store.
            You can retrieve documents based on a query and return the most relevant results.
            The documents will be split into chunks for better indexing and retrieval.
            All your answers will only be based on the indexed content and nothing else. You will not provide any additional information.
            If you don't understand the question or don't have an answer based on the indexed content, you will respond only with the phrase 'I don't know the answer to that question.'

            Question: {question}
            Context: {context}
            Answer:

            """

embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="llama3.2")

# UTILITY FUNCTIONS
# =======================================================================

def load_page(url):
    """Load a web page using Selenium."""
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents

def split_text(documents):
    """Split the text from the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
        )
    
    return text_splitter.split_documents(documents)

def index_docs(documents):
    """Index the documents into the vector store."""
    vectorstore.add_documents(documents)

def retrieve_docs(query):
    """Retrieve documents based on a query."""
    results = vectorstore.similarity_search(query, k=3)
    return results

def answer_question(question, context):
    """Answer a question based on the indexed content."""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer = chain.invoke({"question": question, "context": context})
    return answer

# USER INTERFACE
# =======================================================================

st.title("LLM Web Scraper and Question Answering")
url = st.text_input("Enter the URL to scrape:")

documents = load_page(url)

if documents:
    st.write("Web page loaded successfully.")
    st.write(f"Number of documents loaded: {len(documents)}")
    
    # Split the text into chunks
    chunks = split_text(documents)
    st.write(f"Number of chunks created: {len(chunks)}")
    
    # Index the documents
    index_docs(chunks)
    st.write("Documents indexed successfully.")
    
    question = st.chat_input("Ask a question based on the URL:")
    
    if question:
        st.chat_message("user").write(question)

        results = retrieve_docs(question)
        context = "\n\n".join([doc.page_content for doc in results])
        answer = answer_question(question, context)
        
        st.chat_message("assistant").write(answer)

        if answer != "I don't know the answer to that question.":
            st.info("Context used for the answer:")
            st.markdown(f"""
                            <div style="
                                border: 2px solid #4CAF50;
                                border-radius: 10px;
                                padding: 15px;
                                background-color: #f9f9f9;
                                color: #333;
                                font-size: 16px;
                                line-height: 1.6;
                                ">
                                {context}
                            </div>
                        """, unsafe_allow_html=True)