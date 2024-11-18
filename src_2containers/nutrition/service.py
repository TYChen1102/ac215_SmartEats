from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/v1/data', methods=['GET'])
def get_data():
    data = {"message": "Hello from the backend service!"}
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
