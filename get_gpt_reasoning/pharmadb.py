from datasets import load_dataset
from openai import AzureOpenAI
import pandas as pd
from tqdm.auto import tqdm  # progress bar
import os

# ── 1) Configure Azure OpenAI client ────────────────────────────────────────────
client = AzureOpenAI(
    api_key       = os.getenv("AZURE_OPENAI_API_KEY", "8H18BGFCgtsRenwiUVLicgiRjVS9PmA44cnOpt2OvvxKgRqIbLMtJQQJ99BDACYeBjFXJ3w3AAABACOGLsu4"),  # better via env var
    api_version   = "2024-12-01-preview",
    azure_endpoint= "https://azure-api-jesse.openai.azure.com/"
)

# ── 2) Load the pharmaDB dataset ────────────────────────────────────────────────
ds_train = load_dataset(
    "Tassy24/K-Paths-inductive-reasoning-pharmaDB",
    split="train",
    trust_remote_code=True
)

# Limit to 2 000 randomly‑shuffled examples to keep runtime manageable
sample2k = ds_train if len(ds_train) < 2000 else ds_train.shuffle(seed=42).select(range(2000))

# ── 3) Helper: call GPT‑4o to generate a reasoning path ─────────────────────────

def reasoning_path(drug: str, disease: str, label: str) -> str:
    """Return a free‑text mechanistic reasoning path (does *not* echo the label)."""

    user_prompt = (
        f"Drug: {drug}\n"
        f"Disease: {disease}\n"
        f"Therapeutic relationship (answer): {label}\n\n"
        "Provide a concise and thoughtful step‑by‑step reasoning path that justifies *why* this relationship holds. "
        "Do not restate the label."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical pharmacologist. "
                "Given the answer, explain the mechanistic or clinical evidence in a paragraph."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    resp = client.chat.completions.create(
        model       = "gpt-4o",
        messages    = messages,
        max_tokens  = 150,
        temperature = 0.5,
        top_p       = 1.0
    )

    return resp.choices[0].message.content.strip()


# ── 4) Main loop with live progress bar & checkpointing ─────────────────────────
records = []
partial_path = "gpt4o_reasoning_paths_partial.csv"
final_path   = "gpt4o_reasoning_paths.csv"

for idx, ex in enumerate(tqdm(sample2k, desc="Generating reasoning paths")):
    try:
        rationale = reasoning_path(
            ex["drug_name"],
            ex["disease_name"],
            ex["label"]
        )
    except Exception as err:
        print(f"\n⚠️  Error on row {idx}: {err}. Skipping.")
        rationale = ""

    records.append({
        "drug":        ex["drug_name"],
        "disease":     ex["disease_name"],
        "label":       ex["label"],
        "reason_path": rationale
    })

    # Save a checkpoint every 100 iterations so progress isn't lost.
    if (idx + 1) % 100 == 0:
        pd.DataFrame(records).to_csv(partial_path, index=False)

# ── 5) Final save ───────────────────────────────────────────────────────────────
results = pd.DataFrame(records)
results.to_csv(final_path, index=False)
print(f"\n✅ Completed! Results written to {final_path}")
