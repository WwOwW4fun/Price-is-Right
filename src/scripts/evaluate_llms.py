import os
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.evaluator import evaluate
from litellm import completion
from pricer.items import Item
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
train, val, test = Item.from_hub("ed-donner/items_lite")

def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]

#First model gpt-4.1-nano
def gpt_4__1_nano(item):
    response = completion(model="openai/gpt-4.1-nano", messages=messages_for(item))
    return response.choices[0].message.content

#Second model calude opus 4.5
def claude_opus_4_5(item):
    response = completion(model="anthropic/claude-opus-4-5", messages=messages_for(item))
    return response.choices[0].message.content

#Third model Gemini 3 pro
def gemini_3_pro_preview(item):
    response = completion(model="gemini/gemini-3-pro-preview", messages=messages_for(item), reasoning_effort='low')
    return response.choices[0].message.content

#Fourth model
def gemini_2__5_flash_lite(item):
    response = completion(model="gemini/gemini-2.5-flash-lite", messages=messages_for(item))
    return response.choices[0].message.content

#Fifth model

def grok_4__1_fast(item):
    response = completion(model="xai/grok-4-1-fast-non-reasoning", messages=messages_for(item), seed=42)
    return response.choices[0].message.content


if __name__ == "__main__":
    evaluate(claude_opus_4_5, test)