from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from termcolor import colored
import time

# Empty MPS memory (if using a different backend, e.g., CUDA on Nvidia)
torch.mps.empty_cache()  # This clears MPS memory, if available

model_path = "tmp/Llama-3.2-3B-Instruct/"
peft_model_path = os.path.expanduser("~/PycharmProjects/LetMeFineTune_By_TorchTune/tmp/adapter/")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

model.load_adapter(peft_model_path)

## inference Why sky is blue to this model
model = model.to("mps")

# Load the tokenizer (you should use the same tokenizer used with the base model)
tokenizer = AutoTokenizer.from_pretrained("tmp/Llama-3.2-3B-Instruct")

while True:
    # Take input from the user
    input_text = input(colored("Enter your input text (or type 'exit' to quit): ", "blue"))
    if input_text.lower() == "exit":
        break

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    # Generate text using the model
    print(colored("Generating text...", "green"))
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    end_time = time.time()

    print(colored("Decoding text...", "green"))
    # Decode the generated output back into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(colored(generated_text, "yellow"))
    print(colored(f"Execution time: {end_time - start_time:.2f} seconds", "red"))