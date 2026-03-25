import os
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.items import Item
from tqdm import tqdm
from transformers import AutoTokenizer


load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

dataset = f"lesserafimlover/items"

train, val, test = Item.from_hub(dataset)
items = train + val + test

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

CUTOFF = 110

for item in tqdm(train+val):
    item.make_prompts(tokenizer, CUTOFF, True)
for item in tqdm(test):
    item.make_prompts(tokenizer, CUTOFF, False)

Item.push_prompts_to_hub(dataset, train, val, test)

if __name__ == "__main__":
    print(train[0].prompt)
    print(train[0].completion)