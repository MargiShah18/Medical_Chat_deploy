from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from app import process_and_store_file, augment_prompt, generate_response, setup_file_watcher, process_existing_files

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask Routes
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the RAG Chatbot API!'})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')

    if not query:
        print("❌ No query provided")
        return jsonify({'error': 'No query provided'}), 400

    print(f"✅ Received Query: {query}")

    try:
        augmented_query = augment_prompt(query)
        print(f"🔍 Augmented Query: {augmented_query}")

        response = generate_response(augmented_query)
        print(f"📝 Response Generated: {response}")

        if not response:
            print("❌ Failed to generate response")
            return jsonify({'error': 'Failed to generate response'}), 500

        return jsonify({'response': response, 'augmentedQuery': augmented_query})

    except Exception as e:
        print(f"🔥 Error in processing: {e}")
        return jsonify({'error': str(e)}), 500


# Main entry point
if __name__ == '__main__':
    # Process existing files first
    process_existing_files(UPLOAD_FOLDER)
    
    # Start the file watcher
    observer = setup_file_watcher(UPLOAD_FOLDER)
    try:
        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))
    finally:
        observer.stop()
        observer.join()
