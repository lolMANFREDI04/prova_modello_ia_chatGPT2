import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Controlla se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il tokenizer dal disco locale
tokenizer = GPT2Tokenizer.from_pretrained("./local_model")

# Carica il modello dal disco locale
model = GPT2LMHeadModel.from_pretrained("./local_model")
model.to(device)

# Aggiungi un token di padding al tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Definisci la funzione di tokenizzazione
def tokenize_function(examples):
    # Aggiungi gli spazi per separare il testo target
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs.input_ids.copy()  # Crea i target copiando gli input
    return inputs

# Carica un dataset di esempio per il testing
dataset = load_dataset("ag_news", split='train[:1%]')  # Usa un piccolo dataset per velocizzare il test

# Applica la tokenizzazione al dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Definisci gli argomenti per il training
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Aumenta il batch size se possibile
    save_steps=10_000,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,
)

# Definisci il trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Inizia il training
trainer.train()

# Salva il modello finale
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Training completato e modello salvato.")
