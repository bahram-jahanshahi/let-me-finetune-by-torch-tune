from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Empty MPS memory (if using a different backend, e.g., CUDA on Nvidia)
torch.mps.empty_cache()  # This clears MPS memory, if available

print(torch.backends.mps.is_available())  # Should return True on MPS-supported devices


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = AutoModelForCausalLM.from_pretrained("tmp/merged")
model = model.to(device)

# Load the tokenizer (you should use the same tokenizer used with the base model)
tokenizer = AutoTokenizer.from_pretrained("tmp/Llama-3.2-3B-Instruct")

input_text = "Once upon a time, in a land far, far away"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text using the model
outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

# Decode the generated output back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)


