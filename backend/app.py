from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF for PDF processing
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
PINECONE_API_KEY2 = os.getenv("PINECONE_API_KEY2")

if not OPENAI_API_KEY2 or not PINECONE_API_KEY2:
    raise ValueError("Missing API Keys. Please check your .env file.")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI Chat Model

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY2, model='gpt-3.5-turbo')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY2)
index_name = "rag-chatbot-index"
print(pc.list_indexes().names())
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536)  # Set correct dimension for embeddings
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            process_and_store_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            filename = os.path.basename(event.src_path)
            file_id = str(hash(filename))[:8]  # Generate same file_id as in store_in_pinecone
            delete_file_from_pinecone(file_id)

def process_and_store_file(filepath):
    """Process a single file and store it in Pinecone."""
    try:
        print(f"\n=== Starting to process file: {filepath} ===")
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
            
        if not filepath.lower().endswith('.pdf'):
            print(f"❌ Not a PDF file: {filepath}")
            return
            
        # Extract text from PDF
        text = process_pdf(filepath)
        if not text or not text.strip():
            print("❌ No text extracted from PDF")
            return
            
        filename = os.path.basename(filepath)
        print(f"Extracted text length: {len(text)} characters")
        store_in_pinecone(text, filename)
        print(f"Successfully processed and stored {filename}")
        print("=== File processing completed ===\n")
    except Exception as e:
        print(f"❌ Error processing file {filepath}: {e}")
        raise e

def setup_file_watcher():
    """Set up a file watcher for the upload directory."""
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=UPLOAD_FOLDER, recursive=False)
    observer.start()
    print(f"Started watching directory: {UPLOAD_FOLDER}")
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

def process_existing_files():
    """Process any existing files in the upload folder during startup."""
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
        if files:
            print(f"Found {len(files)} PDF files in upload folder")
            # Clear Pinecone first
            clear_pinecone()
            # Process all files
            for file in files:
                filepath = os.path.join(UPLOAD_FOLDER, file)
                print(f"Processing file: {file}")
                process_and_store_file(filepath)
        else:
            print("No existing PDF files found in upload folder")
            # Clear Pinecone if no files exist
            clear_pinecone()
    except Exception as e:
        print(f"Error processing existing files: {e}")

# Flask Routes
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the RAG Chatbot API!'})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    augmented_query = augment_prompt(query)
    response = generate_response(augmented_query)
    
    if not response:
        return jsonify({'error': 'Failed to generate response'}), 500

    return jsonify({'response': response, 'augmentedQuery': augmented_query})

# Helper Functions
def process_pdf(filepath):
    """Extract text from a PDF."""
    text = ""
    try:
        print(f"Opening PDF file: {filepath}")
        with fitz.open(filepath) as doc:
            print(f"PDF has {len(doc)} pages")
            for page_num, page in enumerate(doc):
                text += page.get_text("text") + "\n"
                print(f"Processed page {page_num + 1}")
        if not text.strip():
            raise ValueError("Extracted text is empty.")
        print(f"Extracted text preview:\n{text[:500]}...")  # Print first 500 characters
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        raise e
    return text

def store_in_pinecone(text, filename):
    """Store extracted text in Pinecone."""
    try:
        print(f"\n=== Starting Pinecone storage for {filename} ===")
        index = pc.Index(index_name, pinecone_api_key=PINECONE_API_KEY2)
        print(f"Connected to Pinecone index: {index_name}")

        # Create a shorter unique identifier for the file
        file_id = str(hash(filename))[:8]  # Use first 8 characters of hash as unique ID
        print(f"Generated file_id: {file_id} for {filename}")

        # Process the current file
        embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY2)
        text_chunks = split_text(text)
        print(f"Split text into {len(text_chunks)} chunks")
        
        print("Generating embeddings...")
        embeddings = embed_model.embed_documents(text_chunks)
        print(f"Generated {len(embeddings)} embeddings")

        print(f"✅ Storing {len(text_chunks)} chunks in Pinecone")
        print(f"First chunk preview: {text_chunks[0][:100]}...")

        # Upload new chunks with fresh embeddings
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
            print(f"📝 Stored chunks {i} to {batch_end-1}")
            
        print(f"=== Pinecone storage completed for file_id {file_id} ===\n")

    except Exception as e:
        print(f"❌ Error during Pinecone upsert: {e}")
        raise e

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

def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def augment_prompt(query):
    """Retrieve relevant context from Pinecone and augment the query."""
    try:
        embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY2)  # Consistent model
        index = pc.Index(index_name,pinecone_api_key=PINECONE_API_KEY2)
        vectorstore = LangchainPinecone(index, embed_model, "text")

        print(f"🔍 Searching Pinecone for query: {query}")  # Debugging Line

        results = vectorstore.similarity_search(query, k=5)

        if not results:
            print("⚠️ No relevant context found in Pinecone.")
            return f"Query: {query}"  # Fallback if no context is found

        # 🔥 FIX: Use doc.page_content instead of metadata['text']
        source_knowledge = "\n".join([doc.page_content for doc in results])

        print(f"✅ Retrieved {len(results)} relevant chunks from Pinecone")  # Debugging Line

        return f"Using the following contexts, answer the query:\n\n{source_knowledge}\n\nQuery: {query}"
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        return query  # Return original query if retrieval fails

def generate_response(augmented_query):
    try:
        prompt = HumanMessage(content=augmented_query)
        res = chat.invoke([SystemMessage(content="You are a helpful assistant."), prompt])
        return res.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Main entry point
if __name__ == '__main__':
    # Process existing files first
    process_existing_files()
    
    # Start the file watcher
    observer = setup_file_watcher()
    try:
        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))
    finally:
        observer.stop()
        observer.join()