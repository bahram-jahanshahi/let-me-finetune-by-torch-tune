from transformers import AutoModelForCausalLM
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained("tmp/Llama-3.2-3B-Instruct/",)

adapter_model = PeftModel.from_pretrained(model, "/Users/bahram/PycharmProjects/LetMeFineTune_By_TorchTune/tmp/adapter", repo_type="local")


# Merge the LoRA weights with the base model
# adapter_model.merge_and_unload()

# Save the merged model
model.save_pretrained("tmp/merged")

