import os
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.schema import HumanMessage, SystemMessage


# Load environment variables
load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
PINECONE_API_KEY2 = os.getenv("PINECONE_API_KEY2")

if not OPENAI_API_KEY2 or not PINECONE_API_KEY2:
    raise ValueError("Missing API Keys. Please check your .env file.")

# Initialize OpenAI Chat Model
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY2, model='gpt-3.5-turbo')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY2)
index_name = "rag-chatbot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536)  # Set correct dimension for embeddings
else:
    print(f"Index '{index_name}' already exists.")

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_and_store_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            file_id = str(hash(filename))[:8]  # Generate same file_id as in store_in_pinecone
            delete_file_from_pinecone(file_id)

def delete_file_from_pinecone(file_id):
    """Delete vectors for a specific file."""
    try:
        index = pc.Index(index_name, pinecone_api_key=PINECONE_API_KEY2)
        
        # For Serverless/Starter indexes, we need to delete all and re-upload remaining files
        print("🧹 Clearing all vectors...")
        index.delete(delete_all=True)
        print("✅ All vectors cleared")

        # Re-upload all remaining files except the one being deleted
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
        print(f"Re-uploading {len(files)} remaining files...")
        embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY2)
        
        for f in files:
            current_id = str(hash(f))[:8]
            if current_id != file_id:  # Skip the file we're deleting
                try:
                    filepath = os.path.join(UPLOAD_FOLDER, f)
                    print(f"Processing file: {f}")
                    with fitz.open(filepath) as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text("text") + "\n"
                        
                        if text.strip():
                            chunks = split_text(text)
                            embeddings = embed_model.embed_documents(chunks)
                            
                            print(f"Re-uploading chunks for {f}")
                            batch_size = 10
                            for i in range(0, len(chunks), batch_size):
                                batch_end = min(i + batch_size, len(chunks))
                                batch = []
                                for j in range(i, batch_end):
                                    doc_id = f"{current_id}-{j}"
                                    metadata = {
                                        'text': chunks[j],
                                        'source': f,
                                        'file_id': current_id,
                                        'chunk_id': j
                                    }
                                    batch.append((doc_id, embeddings[j], metadata))
                                index.upsert(vectors=batch)
                except Exception as e:
                    print(f"⚠️ Error re-uploading file {f}: {e}")
                    
        print("✅ Finished re-uploading remaining files")
    except Exception as e:
        print(f"❌ Error during vector deletion: {e}")
        raise e


def process_and_store_file(filepath):
    """Process a single file and store it in Pinecone."""
    try:
        if not os.path.exists(filepath):
            return
        if not filepath.lower().endswith('.pdf'):
            return
        text = process_pdf(filepath)
        if not text.strip():
            return
        filename = os.path.basename(filepath)
        store_in_pinecone(text, filename)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def process_pdf(filepath):
    """Extract text from a PDF."""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        if not text.strip():
            raise ValueError("Extracted text is empty.")
    except Exception as e:
        raise e
    return text

def store_in_pinecone(text, filename):
    """Store extracted text in Pinecone."""
    try:
        index = pc.Index(index_name, pinecone_api_key=PINECONE_API_KEY2)
        file_id = str(hash(filename))[:8]
        embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY2)
        text_chunks = split_text(text)
        embeddings = embed_model.embed_documents(text_chunks)
        batch_size = 10
        for i in range(0, len(text_chunks), batch_size):
            batch_end = min(i + batch_size, len(text_chunks))
            batch = []
            for j in range(i, batch_end):
                doc_id = f"{file_id}-{j}"
                metadata = {
                    'text': text_chunks[j],
                    'source': filename,
                    'file_id': file_id,
                    'chunk_id': j
                }
                batch.append((doc_id, embeddings[j], metadata))
            index.upsert(vectors=batch)
    except Exception as e:
        raise e

def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def augment_prompt(query):
    """Retrieve relevant context from Pinecone and augment the query."""
    try:
        embed_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY2)
        index = pc.Index(index_name, pinecone_api_key=PINECONE_API_KEY2)
        vectorstore = LangchainPinecone(index, embed_model, "text")

        results = vectorstore.similarity_search(query, k=5)
        if not results:
            return f"Query: {query}"

        source_knowledge = "\n".join([doc.page_content for doc in results])
        return f"Using the following contexts, answer the query:\n\n{source_knowledge}\n\nQuery: {query}"
    except Exception as e:
        return query


def generate_response(augmented_query):
    try:
        prompt = HumanMessage(content=augmented_query)
        res = chat.invoke([SystemMessage(content="You are a helpful assistant."), prompt])
        return res.content
    except Exception as e:
        print(f"Error generating response 1111: {e}")
        return None


def setup_file_watcher(upload_folder):
    """Set up a file watcher for the upload directory."""
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=upload_folder, recursive=False)
    observer.start()
    return observer

def clear_pinecone():
    """Clear all vectors from Pinecone."""
    try:
        index = pc.Index(index_name, pinecone_api_key=PINECONE_API_KEY2)
        print("🧹 Clearing all vectors from Pinecone...")
        index.delete(delete_all=True)
        print("✅ All vectors cleared from Pinecone")
    except Exception as e:
        print(f"❌ Error clearing Pinecone: {e}")
        raise e


def process_existing_files(upload_folder):
    """Process any existing files in the upload folder during startup."""
    try:
        files = [f for f in os.listdir(upload_folder) if f.endswith('.pdf')]
        for file in files:
            filepath = os.path.join(upload_folder, file)
            process_and_store_file(filepath)
    except Exception as e:
        print(f"Error processing existing files: {e}")

