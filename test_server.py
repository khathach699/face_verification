from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    print("Received POST request to /verify")
    print("Request data:", request.json)
    return jsonify({'status': 'ok', 'message': 'Test successful'})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    print("Starting test Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
