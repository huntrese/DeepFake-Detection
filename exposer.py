# !pip install pyngrok

from flask import Flask, request, send_file
from pyngrok import ngrok

# Set up Flask app
app = Flask(__name__)
ngrok.set_auth_token("2aUKxhETRkeaHgT49YAsp3KlWpc_3WYPatDEW3zozbZzFdw8A")

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.json.get('prompt', '')
        if prompt:
            return "Waau"
        else:
            return "No prompt provided", 400
    else:
        return "Please send a POST request with a prompt", 200

# Set up ngrok
port_no = 5000
public_url = ngrok.connect(port_no).public_url
print(f"To access the global link, please click {public_url}")

# Run the app
app.run(port=port_no)