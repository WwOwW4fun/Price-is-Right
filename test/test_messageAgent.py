from agents.messaging_agent import MessagingAgent

agent = MessagingAgent()
if __name__ == "__main__":
    #agent.push("Hi")
    agent.notify("A special deal on Sumsung 60 inch LED TV going at a great bargain", 300, 1000, "www.samsung.com")