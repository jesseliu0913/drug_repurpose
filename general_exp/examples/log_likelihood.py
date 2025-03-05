import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

question = "What is the capital of the UK?"
target_answer = " London"

input_text = f"Question: {question}\nAnswer:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
answer_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[:, -answer_ids.shape[1] - 1:-1, :]

log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
answer_log_probs = log_probs.gather(2, answer_ids.unsqueeze(2)).squeeze(2)
log_likelihood = answer_log_probs.sum().item()

decoded_tokens = [tokenizer.decode(token) for token in answer_ids.squeeze().tolist()]

print(f"Log-Likelihood of the answer '{target_answer.strip()}': {log_likelihood}")
print("Decoded Tokens from the Model Output:")
for token, log_prob in zip(decoded_tokens, answer_log_probs.squeeze().tolist()):
    print(f"Token: '{token}', Log-Probability: {log_prob:.4f}")
# CUDA_VISIBLE_DEVICES=7