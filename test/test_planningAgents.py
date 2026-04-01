from dotenv import load_dotenv
import chromadb
import logging

root = logging.getLogger()
root.setLevel(logging.INFO)

#Loading database
DB = "products_vectorstore"
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')
from agents.autonomous_planning_agent import AutonomousPlanningAgent
agent = AutonomousPlanningAgent(collection)

if __name__ == "__main__":
    agent.plan()