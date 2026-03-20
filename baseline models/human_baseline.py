from pricer.evaluator import evaluate
from pricer.items import Item
import csv

train, val, test = Item.from_hub("ed-donner/items_lite")

# Human Baseline 
# Write the test set to a CSV

with open('data/raw/human_in.csv', 'w', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:100]:
        writer.writerow([t.summary, 0])

# Read it back in

human_predictions = []
with open('data/raw/human_out.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))

def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]
if __name__ == "__main__":
    evaluate(human_pricer, test, size=100)