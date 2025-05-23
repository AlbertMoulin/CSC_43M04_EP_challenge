from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-2b"  # or whichever Gemma variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt")

# Get outputs with hidden states
outputs = model(**inputs, output_hidden_states=True)

# The last hidden state (before output projection)
last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

# If you want just the last token's representation
last_token_hidden = last_hidden_state[:, -1, :]  # Shape: [batch_size, hidden_size]