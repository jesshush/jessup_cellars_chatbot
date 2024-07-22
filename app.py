from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response  

app = Flask(__name__)
CORS(app)

# Initialize conversation context
context = {'last_response': None, 'last_tag': None}

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    global context 
    try:
       
        data = request.get_json()
        text = data.get("message")
        
        if text:
           
            if text.lower() == "tell me more about it":
                if context['last_response']:
                    response, tag = get_response(context['last_response'], context)
                else:
                    response = "I'm sorry, but I don't have any previous context to expand on. Could you please ask a specific question?"
                    tag = None
            else:
                response, tag = get_response(text)
                context['last_response'] = response
                context['last_tag'] = tag

           
            message = {"answer": response}
            return jsonify(message)
        else:
            
            return jsonify({"error": "No message provided"}), 400
    except Exception as e:
    
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
