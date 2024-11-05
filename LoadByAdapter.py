from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "tmp/Llama-3.2-3B-Instruct/"
peft_model_path = "/Users/bahram/PycharmProjects/LetMeFineTune_By_TorchTune/tmp/adapter/"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.load_adapter(peft_model_path)

model = model.to("mps")

# Load the tokenizer (you should use the same tokenizer used with the base model)
tokenizer = AutoTokenizer.from_pretrained("tmp/Llama-3.2-3B-Instruct")

input_text = "Once upon a time, in a land far, far away"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to("mps")

# Generate text using the model
print("Generating text...")
outputs = model.generate(**inputs, max_length=20, num_return_sequences=1)

print("Decoding text...")
# Decode the generated output back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)