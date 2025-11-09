from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/process', methods=['POST'])
def greet():
    try:
        data = request.get_json()
        # Process your data here

        # Example response
        name = data.get('name', 'Guest')
        return jsonify({
            'message': f'Hello from backend 👋 {name} !'
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)