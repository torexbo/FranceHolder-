from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-Neo model (or GPT-2 as an alternative)
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/respond', methods=['POST'])
def respond():
    user_message = request.json['message']
    
    # Tokenize the user input and generate a response
    inputs = tokenizer.encode(user_message, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
