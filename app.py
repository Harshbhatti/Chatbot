import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# Initialize model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # Use medium for faster responses
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Predefined responses with links
predefined_responses = {
    "services": "At Easy Link Web & IT Solutions, we offer services like web design & development, digital marketing, Google Workspace, cloud hosting, and cybersecurity.",
    "web design": "Our web design services include both frontend and backend development. Learn more here: https://www.easylinkindia.com/web-design.php",
    "digital marketing": "We provide digital marketing solutions to boost your brand. Visit: https://www.easylinkindia.com/digital-marketing.php",
    "google workspace": "We are certified in Google Workspace deployment. More details: https://www.easylinkindia.com/workspace.php",
    "cloud hosting": "Our cloud hosting solutions are secure and scalable. Check here: https://www.easylinkindia.com/cloud-hosting-and-storage.php",
    "cybersecurity": "We offer comprehensive cybersecurity services. Find out more: https://www.easylinkindia.com/cybersecurity.php",
    "ecommerce solutions": "We provide end-to-end eCommerce solutions, from setup to optimization, ensuring a smooth user experience. More info: https://www.easylinkindia.com/ecommerce.php",
    "mobile apps": "We develop mobile and web-based applications tailored to meet specific business needs. Visit: https://www.easylinkindia.com/mobile-app.php",
    "seo services": "We provide SEO and SEM services to improve your website’s visibility and search rankings. Details here: https://www.easylinkindia.com/seo.php",
    "content marketing": "Our content marketing services help in creating engaging content to enhance brand presence. Learn more: https://www.easylinkindia.com/content-marketing.php",
    "contact": "Contact us at WorkFlo Greeta Towers, Industrial Estate, Perungudi, OMR Phase 1, Chennai, Tamil Nadu - 600096. Call +91 9585182141 or email info@easylinkindia.com."
}

# Flask app initialization
app = Flask(__name__)

# Generate response function
def generate_response(user_input):
    # Check for predefined responses
    for keyword, response in predefined_responses.items():
        if keyword in user_input.lower():
            return response
    
    # Generate custom response using DialoGPT
    inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors="pt", padding=True).to(device)
    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None

    reply_ids = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.4,
        top_k=50,
        do_sample=True
    )

    return tokenizer.decode(reply_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# Flask route to handle chatbot interaction
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        # Get user input from query parameters (e.g., /chat?message=hi)
        user_input = request.args.get('message', '')
    else:  # POST request
        user_input = request.json.get('user_input', '')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Check for predefined options (1 to 10)
    option_map = {
        "1": "services",
        "2": "web design",
        "3": "google workspace",
        "4": "cloud hosting",
        "5": "contact",
        "6": "ecommerce solutions",
        "7": "mobile apps",
        "8": "seo services",
        "9": "content marketing",
        "10": "cybersecurity"
    }

    if user_input in option_map:
        bot_response = predefined_responses[option_map[user_input]]
    else:
        bot_response = generate_response(user_input)

    return jsonify({"response": bot_response})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)