from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the GPT-Neo model and tokenizer from Hugging Face
model_name = "EleutherAI/gpt-neo-125M"  # You can also choose a larger model if needed
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define chatbot functionality
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question', '')

    # Customize the prompt with Kaggle-specific information
    prompt = f"Answer the following question about Kaggle: {user_input}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
