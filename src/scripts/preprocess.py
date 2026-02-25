import os
from litellm import completion
from dotenv import load_dotenv
import json
from pricer.items import Item
from groq import Groq

load_dotenv(override=True)

#load the raw dataset from ed-donner
username = "ed-donner"
dataset = f"{username}/items_raw_lite"
train, val, test = Item.from_hub(dataset)

items = train + val + test

# Give every item an id

for index, item in enumerate(items):
    item.id = index
# System prompt
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

#Build the JSONL Batch file 
def make_jsonl(item):
    body = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": item.full}
        ]
    }
    line = {
        "custom_id": str(item.id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }
    return json.dumps(line)

def make_file(start, end, filename):
    with open(filename, "w") as f:
        for i in range(start, end):
            f.write(make_jsonl(items[i]))
            f.write("\n")

#initialize groq and make file
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
with open("data/raw/processed/0_1000.jsonl", "rb") as f:
    response = groq.files.create(file=f, purpose="batch")

#send the batch file to groq and retrieve
file_id = response.id
response = groq.batches.create(completion_window="24h", endpoint="/v1/chat/completions", input_file_id=file_id)
result = groq.batches.retrieve(response.id)
# collect the result 
#response = groq.files.content(result.output_file_id)
#response.write_to_file("data/raw/processed/batch_results.jsonl")

with open("data/raw/processed/batch_results.jsonl", "r") as f:
    for line in f:
        json_line = json.loads(line)
        id = int(json_line["custom_id"])
        summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
        items[id].summary = summary
                                                    
#Fix the HF error 
def normalize_for_hub(item: Item) -> Item:
    # Ensure optional strings are always strings (not None)
    if item.full is None:
        item.full = ""
    if item.summary is None:
        item.summary = ""
    if item.prompt is None:
        item.prompt = ""
    return item

items = [normalize_for_hub(it) for it in items]

#push the data to hugging face
#train = items[:20_000]
#val = items[20_000:21_000]
#test = items[21_000:]
#Item.push_to_hub("lesserafimlover/items_lite", train, val, test)

# if __name__ == "__main__":
#     print(len(items))
