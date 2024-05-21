import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carica il tokenizer e il modello fine-tuned dal disco locale
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

# Assicurati che il modello sia in modalità di valutazione
model.eval()

# Controlla se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definisci il prompt
prompt = "Money Funds Fell in Latest Week"

# Tokenizza il prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Genera il testo
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    top_p=0.9,
    do_sample=True,
    pad_token_id=50256
)

# Decodifica il testo generato
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Testo generato:")
print(generated_text)
