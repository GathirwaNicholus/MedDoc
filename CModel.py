from flask import Flask, render_template, request, jsonify, session
from MedDoc import preprocess_input, BertModel, train_model, BertTokenizer
import timeit
import torch
from llama_cpp import Llama
import threading
import MedDoc
import uuid
# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# UI stuff from here

model_path = "capybarahermes-2.5-mistral-7b.Q5_K_M.gguf"

# Disabling gradient tracking
torch.no_grad()

# Load llama2 model
llm = Llama(model_path=model_path,
            n_ctx=512,
            n_batch=128,)

# Start timer
start = timeit.default_timer()

max_length = 1500  # Adjust maximum output length

# Pre-prompt to focus on heart health
pre_prompt = "Let's talk about your heart health. Ask me anything related to heart health, symptoms, or conditions. "


app = Flask(__name__, static_folder='static', template_folder='templates')


app.secret_key = "58584983098739299398832"

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('login.html', user_id=session['user_id'])

@app.route('/trying')
def trying():
    return render_template('generate.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Get user prompt from the request data
        prompt = request.get_json()['prompt']
        pre_processed_input = preprocess_input(prompt)
        input_text = prompt
        print(f"What gets sent to MedDoc looks like this: {pre_processed_input}")

        # Check if prompt mentions heart-related keywords (case-insensitive)
        keywords = ["heart", "health", "angina", "cholesterol", "age"]
        if not any(word in prompt.lower() for word in keywords):
            print(f"Hello, While that might be interesting, what specific heart health questions do you have? ")
            # Since the prompt lacks keywords, return without MedDoc processing
            return jsonify({'text': "Please enter a question related to your heart health.", 'user_id': session['user_id'], 'congrats_message': '', 'sad_message': ''})

        encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

        # Pass the input through the BERT model
        outputs = bert_model(**encoded_input)

        # Extract the pooled output (CLS token) or the hidden states as per your requirement
        pooled_output = outputs.pooler_output  # Example: using the pooled output

        from MedDoc import classifier
        # Pass the pooled_output to the classifier for prediction
        logits = classifier(pooled_output)

        import torch.nn as nn
        # Apply softmax to obtain probabilities
        probs = nn.functional.softmax(logits, dim=-1)

        # Obtain the predicted class
        predicted_class = torch.argmax(probs)

        if predicted_class == 0:
            output = 0
            sad_message = "Great news! Based on the information you've provided, it seems unlikely you have heart disease. However, it's always recommended to consult a doctor for a professional diagnosis."
        else:
            output = 1
            congrats_message = "I understand this might be concerning. While the model suggests a low probability of heart disease, it's important to remember this is for informational purposes only. Please don't hesitate to reach out to a medical professional for a proper diagnosis and guidance."

        if output == 1:
            pre_prompt = "I have heart disease"
        else:
            pre_prompt = "I do not have heart disease"
        print(f"The output for MedDoc is {output}")

        mOutput = output

        post_prompt = "How can you help me diagnose the issue? What is the way forward for me?"
        prompt = pre_prompt + prompt + post_prompt

        # Generate text incrementally (replace print with appending to generated_text)
        generated_text = ""
        while len(generated_text) < max_length and not generated_text.endswith("."):
            # Generate a batch of tokens
            output = llm(prompt, max_tokens=5, echo=False, temperature=0.1, top_p=0.9)
            next_token = output['choices'][0]['text']

            # Update generated text and write it to file
            generated_text += next_token
            with open("conversation.txt", "a") as f:
                f.write(f"{session['user_id']}: {input_text}\n")
                f.write(f"{next_token}")
                f.write("\n")
                f.write(f"Input for MedDoc model is {pre_processed_input}")
                f.write(f"Output for MedDoc model is {mOutput}")

            print(f"Generated Text: {generated_text}")

            # Update prompt for next iteration
            prompt = generated_text

        # No need to save to database since we're using a text file

        return jsonify({'text': generated_text, 'user_id': session['user_id'], 'congrats_message': congrats_message if output == 1 else sad_message})


    
if __name__ == '__main__':
       app.run(debug=True)

