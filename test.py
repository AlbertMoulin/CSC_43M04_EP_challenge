# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "google/gemma-3-1b-it"  # or whichever Gemma variant
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# text = "Your input text here"
# inputs = tokenizer(text, return_tensors="pt")
# print("inputs")
# print(inputs)
# # Get outputs with hidden states
# outputs = model(**inputs, output_hidden_states=True)
# print("outputs")
# print(outputs)
# # The last hidden state (before output projection)
# last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
# print("last_hidden_state")
# print(last_hidden_state.shape)  # Check the shape
# print(last_hidden_state)
# # If you want just the last token's representation
# last_token_hidden = last_hidden_state[:, -1, :]  # Shape: [batch_size, hidden_size]
# print("last_token_hidden")
# print(last_token_hidden)


### Test if the model is actually prediction something and not just predicting the mean
import pandas as pd
pd.read_csv("submission.csv").head()  # Check the submission file format

# make histogram of the predictions
import matplotlib.pyplot as plt
predictions = pd.read_csv("submission.csv")["views"].values
plt.hist(predictions, bins=50)
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Views")
plt.ylabel("Frequency")
plt.savefig('myplot.png')
print(f"Min prediction: {predictions.min():,.0f}")
print(f"Max prediction: {predictions.max():,.0f}")
print(f"Mean prediction: {predictions.mean():,.0f}")
print(f"Std deviation: {predictions.std():,.0f}")
print(f"Unique values: {len(set(predictions))}")
print(f"total predictions: {len(predictions)}")

# plot training data distribution of views
train_data = pd.read_csv("dataset/train_val.csv")
plt.hist(train_data["views"], bins=50)
plt.title("Distribution of Training Views")
plt.xlabel("Views")
plt.ylabel("Frequency")
plt.savefig('train_data_distribution.png')
