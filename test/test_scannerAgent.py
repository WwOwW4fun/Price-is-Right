import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.deals import ScrapedDeal, DealSelection
import logging
import requests
from agents.scanner_agent import ScannerAgent

load_dotenv(override=True)
openai = OpenAI()
MODEL = 'gpt-5-mini'

agent = ScannerAgent()
result = agent.scan()

if __name__ == "__main__":
    print(result)