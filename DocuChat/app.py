import streamlit as st
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
import shutil
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import zipfile
import base64


st.set_page_config(layout="wide")

# Initialize session state with default values
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.documents_processed = False
    st.session_state.processing_complete = False
    st.session_state.files_uploaded = False
    st.session_state.vectorstore_path = None

# Initialize user_name if not already set
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Firebase initialization
@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("/content/drive/MyDrive/Colab Notebooks/DocChat/document-chatbot-generat-bf086-firebase-adminsdk-5vesf-fec868f7e2.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'document-chatbot-generat-bf086.appspot.com'
        })
    return firestore.client(), storage.bucket()

try:
    db, bucket = initialize_firebase()
except Exception as e:
    st.error(f"Failed to initialize Firebase: {e}")

# Create necessary directories
local_directory = "data"
output_directory = "output"
os.makedirs(local_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

def create_download_link(file_path, link_text):
    """Create a download link for a file"""
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    filename = os.path.basename(file_path)
    mime_type = 'application/zip' if file_path.endswith('.zip') else 'application/octet-stream'
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def clean_directory(directory):
    """Safely clean a directory by removing all files and subdirectories"""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                st.warning(f"Failed to remove {item_path}: {e}")

def load_document(file_path):
    """Load document based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()
        
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()
        
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            text_content = []
            for index, row in df.iterrows():
                row_text = f"Row {index + 1}:\n"
                for column in df.columns:
                    row_text += f"{column}: {row[column]}\n"
                text_content.append({"page_content": row_text, "metadata": {"source": file_path}})
            return text_content
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def save_vectorstore():
    """Save the vectorstore to disk"""
    if st.session_state.vectorstore:
        vectorstore_path = os.path.join(output_directory, "vectorstore")
        os.makedirs(vectorstore_path, exist_ok=True)
        st.session_state.vectorstore.save_local(vectorstore_path)
        st.session_state.vectorstore_path = vectorstore_path
        return vectorstore_path
    return None

def create_standalone_chatbot():
    """Create a standalone chatbot script"""
    standalone_code = '''
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():

    GROQ_API_KEY = "<Your-API-KEY-here>"
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    # Load the saved vectorstore
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.load_local("vectorstore", embeddings)
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

# Main UI
st.title("Document Chatbot")

# Initialize the chatbot
qa_chain = initialize_chatbot()

# Chat interface
query = st.text_input("Ask your question:")

if st.button("Ask", disabled=not query):
    if query:
        try:
            with st.spinner("Generating answer..."):
                response = qa_chain.invoke({"query": query})
                answer = response.get("result", "No response generated.")
                st.write("### Answer")
                st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please enter a question.")
'''
    
    script_path = os.path.join(output_directory, "chatbot.py")
    with open(script_path, "w") as f:
        f.write(standalone_code)
    return script_path

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = '''
streamlit
langchain
faiss-cpu
sentence-transformers
langchain-community==0.2.15
langchain-chroma==0.1.3
langchain-text-splitters==0.2.2
langchain-huggingface==0.0.3
langchain-groq==0.1.9
unstructured==0.15.0
unstructured[pdf]==0.15.0
nltk==3.8.1
'''
    requirements_path = os.path.join(output_directory, "requirements.txt")
    with open(requirements_path, "w") as f:
        f.write(requirements.strip())
    return requirements_path

def create_downloadable_package():
    """Create a downloadable zip package with all necessary files"""
    try:
        # Save vectorstore
        vectorstore_path = save_vectorstore()
        
        # Create standalone chatbot script
        script_path = create_standalone_chatbot()
        
        # Create requirements file
        requirements_path = create_requirements_file()
        
        # Create README
        readme_content = '''
# Standalone Document Chatbot

## Setup Instructions
1. Extract all files to a directory
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. add your own groq API KEY in the code.

4. Run the chatbot:
   ```
   streamlit run chatbot.py
   ```

'''
        readme_path = os.path.join(output_directory, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content.strip())
        
        # Create zip file
        zip_path = os.path.join(output_directory, "chatbot_package.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(script_path, os.path.basename(script_path))
            zipf.write(requirements_path, os.path.basename(requirements_path))
            zipf.write(readme_path, os.path.basename(readme_path))
            
            # Add vectorstore files
            for root, _, files in os.walk(vectorstore_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("vectorstore", os.path.relpath(file_path, vectorstore_path))
                    zipf.write(file_path, arcname)
        
        return zip_path
    except Exception as e:
        st.error(f"Error creating downloadable package: {e}")
        return None

def process_documents():
    with st.spinner("Processing documents..."):
        try:
            documents = []
            
            # Process each file in the directory
            for filename in os.listdir(local_directory):
                file_path = os.path.join(local_directory, filename)
                if os.path.isfile(file_path):  # Only process files
                    doc_content = load_document(file_path)
                    if doc_content:
                        documents.extend(doc_content)
            
            if not documents:
                st.error("No content could be extracted from the documents.")
                return False
                
            # Split documents
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            text_chunks = text_splitter.split_documents(documents)
            
            if not text_chunks:
                st.error("No text chunks were created from the documents.")
                return False

            # Create embeddings
            embedding = HuggingFaceEmbeddings()
            
            # Create vectorstore
            vectorstore = FAISS.from_documents(text_chunks, embedding)
            st.session_state.vectorstore = vectorstore
            
            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            st.session_state.qa_chain = qa_chain
            st.session_state.documents_processed = True
            st.session_state.processing_complete = True
            
            return True
            
        except Exception as e:
            st.error(f"Error during document processing: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return False

def upload_to_firebase(file_path, file_name, user_name):
    """Upload file to Firebase storage in user-specific folders"""
    try:
        blob = bucket.blob(f"{user_name}/uploaded_files/{file_name}")
        blob.upload_from_filename(file_path)
        file_url = blob.public_url
        return file_url
    except Exception as e:
        st.error(f"Error uploading file to Firebase: {e}")
        return None

# Main UI
if not st.session_state.user_name:
    st.session_state.user_name = st.text_input("Please enter your name:")
    if st.button("Submit", disabled=not st.session_state.user_name):
        if not st.session_state.user_name:
            st.warning("Please enter your name")
        if st.session_state.user_name:
            st.session_state.initialized = True
else:
    st.sidebar.title(f"Welcome, {st.session_state.user_name}!")

    # Sidebar for file upload
    st.sidebar.title("File Upload")

    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'csv'],
        accept_multiple_files=True
    )

    if uploaded_files:
        try:
            # Clean the directory before uploading new files
            clean_directory(local_directory)
            
            # Save and upload new files
            for file in uploaded_files:
                file_path = os.path.join(local_directory, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Upload to Firebase
                file_url = upload_to_firebase(file_path, file.name, st.session_state.user_name)
                if file_url:
                    st.session_state.files_uploaded = True
                    st.sidebar.success(f"File uploaded successfully: {file.name}")
                else:
                    st.sidebar.error(f"Failed to upload {file.name} to Firebase.")
            
        except Exception as e:
            st.sidebar.error(f"Error handling files: {e}")

    # Process Documents button
    if st.sidebar.button("Process Documents", disabled=not st.session_state.files_uploaded):
        st.session_state.processing_complete = False
        success = process_documents()
        if success:
            st.success("Documents processed successfully!")
        else:
            st.error("Failed to process documents. Please check the errors above.")

    # Main chat interface
    st.title("Document Chatbot")

    # Create two columns for chat and download section
    chat_col, download_col = st.columns([2, 1])

    with chat_col:
        if st.session_state.documents_processed and st.session_state.processing_complete:
            query = st.text_input("Ask your question about the documents:")
            
            if st.button("Ask", disabled=not query):
                if not query:
                    st.warning("Please enter a question.")
                else:
                    try:
                        with st.spinner("Generating answer..."):
                            response = st.session_state.qa_chain.invoke({"query": query})
                            answer = response.get("result", "No response generated.")
                            st.session_state.chat_history.append({"question": query, "answer": answer})
                            
                            # Display the latest answer immediately
                            st.write("### Latest Answer")
                            st.write(f"**Question:** {query}")
                            st.write(f"**Answer:** {answer}")
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")

            # Display chat history
            if st.session_state.chat_history:
                st.write("### Previous Chat History")
                for chat in reversed(st.session_state.chat_history[:-1]):
                    with st.container():
                        st.write(f"**Question:** {chat['question']}")
                        st.write(f"**Answer:** {chat['answer']}")
                        st.divider()
        else:
            st.info("Please upload and process documents to start chatting.")

    # Debug information
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("### Debug Information")
        st.sidebar.write("Files uploaded:", st.session_state.files_uploaded)
        st.sidebar.write("Documents processed:", st.session_state.documents_processed)
        st.sidebar.write("Processing complete:", st.session_state.processing_complete)
        st.sidebar.write("Vectorstore initialized:", st.session_state.vectorstore is not None)
        st.sidebar.write("QA Chain initialized:", st.session_state.qa_chain is not None)
        if os.path.exists(local_directory):
            st.sidebar.write("Files in local directory:", os.listdir(local_directory))

    # Download section in the right column
    with download_col:
        if st.session_state.documents_processed and st.session_state.processing_complete:
            st.write("### Download Chatbot")
            st.write("Download a standalone version of the chatbot with your processed documents.")
            
            if st.button("Generate Download Package"):
                with st.spinner("Creating downloadable package..."):
                    try:
                        zip_path = create_downloadable_package()
                        if zip_path and os.path.exists(zip_path):
                            # Create download link
                            download_link = create_download_link(
                                zip_path,
                                "Download Chatbot Package"
                            )
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            st.success("""
                            Package created successfully! The download package includes:
                            - Standalone chatbot script
                            - Processed document embeddings
                            - Requirements file
                            - Setup instructions
                            """)
                            
                            # Upload package to Firebase
                            package_url = upload_to_firebase(
                                zip_path,
                                f"chatbot_package_{st.session_state.user_name}.zip",
                                st.session_state.user_name
                            )
                        else:
                            st.error("Failed to create download package.")
                    except Exception as e:
                        st.error(f"Error creating download package: {e}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.info("Process documents first to enable download options.")

# Cleanup on session end
def cleanup():
    """Clean up temporary files when the session ends"""
    try:
        clean_directory(local_directory)
        clean_directory(output_directory)
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

# Register the cleanup function to run when the session ends
st.session_state['_cleanup'] = cleanup
