import os
from litellm import completion
from dotenv import load_dotenv
import json
from pricer.items import Item
from pricer.batch import Batch
from groq import Groq
import time

load_dotenv(override=True)

#load the raw dataset from ed-donner
username = "ed-donner"
dataset = f"{username}/items_raw_lite"
train, val, test = Item.from_hub(dataset)

items = train + val + test
#give every item an id
for index, item in enumerate(items):
   item.id = index
LITE_MODE = True 

Batch.create(items, LITE_MODE)
Batch.run()

#waiting before fetch
MAX_WAIT_MINUTES = 60
POLL_INTERVAL_SECONDS = 30
waited = 0

while waited < MAX_WAIT_MINUTES * 60:
    Batch.fetch()
    
    finished = len([b for b in Batch.batches if b.done])
    total = len(Batch.batches)
    print(f"Progress: {finished}/{total} batches complete")
    
    if finished == total:
        print("All batches complete!")
        break
        
    print(f"Waiting {POLL_INTERVAL_SECONDS}s before checking again...")
    time.sleep(POLL_INTERVAL_SECONDS)
    waited += POLL_INTERVAL_SECONDS
else:
    print("Timed out waiting. Run fetch_only.py to resume later.")
    sys.exit(1)

#remove unnecessary 
for item in items:
    item.full = None
    item.id = None
#push the items to hugging face
train = items[:20_000]
val   = items[20_000:21_000]
test  = items[21_000:]
Item.push_to_hub(f"lesserafimlover/items_lite", train, val, test)

if __name__ == "__main__":
   missing = [item for item in items if not item.summary]
   print(f"Items with missing summaries: {len(missing)}") 