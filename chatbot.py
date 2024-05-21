import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carica il tokenizer dal disco locale
tokenizer = GPT2Tokenizer.from_pretrained("./local_model")

# Carica il modello dal disco locale
model = GPT2LMHeadModel.from_pretrained("./local_model")


def generate_response(input_text, model, tokenizer, max_length=50):
    # Tokenizza il testo di input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Genera la risposta
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decodifica il testo generato
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Testare il chatbot
if __name__ == "__main__":
    input_text = "Ciao, come stai?"
    response = generate_response(input_text, model, tokenizer)
    print(f"Chatbot: {response}")

    while True:
        input_text = input("You: ")
        if input_text.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Arrivederci!")
            break
        
        response = generate_response(input_text, model, tokenizer)
        print(f"Chatbot: {response}")
