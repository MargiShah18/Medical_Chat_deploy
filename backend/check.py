from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins, methods, and headers for testing purposes

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({'message': 'Hello, Postman!'}), 200

@app.route('/api/echo', methods=['POST'])
def echo_endpoint():
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    return jsonify({'received': data}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

