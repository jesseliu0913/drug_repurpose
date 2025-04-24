import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gpt(prompt):
    response_dict = {}
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Once upon a time,",
        max_tokens=5,
        temperature=0.0,
        logprobs=5
    )
    output = response.choices[0].text
    logp = response.choices[0].logprobs

    # for token, lp, topk in zip(logp.tokens, logp.token_logprobs, logp.top_logprobs):
    #     print(f"Token: {token!r}")
    #     print(f"  → logprob: {lp:.4f}")
    #     print("  → top alternatives:")
    #     for alt_tok, alt_lp in topk.items():
    #         print(f"      {alt_tok!r}: {alt_lp:.4f}")
    #     print()
    
    response_dict["output"] = output
    response_dict["logprobs"] = logp
    return response_dict

prompt = "Once upon a time,"
response_dict = call_gpt(prompt)
print(response_dict)
