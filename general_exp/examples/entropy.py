from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "One answer to the question What is the national instrument of Ireland? is Fiddle. Another answer is (Directly told me the answer without any other words)"

num_smaples = 10
answer_lst = []
for _ in range(num_smaples):
    output = text_generator(
        prompt, 
        max_new_tokens=5, 
        num_return_sequences=1, 
        temperature=0.7,  
        top_k=50,         
        top_p=0.9,       
        do_sample=True    
    )[0]['generated_text']
    answer = output.replace(prompt, "").strip()
    answer_lst.append(answer)
print(answer_lst)
# for seq in sequences:
#     print(seq['generated_text'])
