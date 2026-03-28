import os
import locale
import modal
from agents.preprocessor import Preprocessor
from dotenv import load_dotenv
load_dotenv(override=True)
from support.pricer_ephemeral import app, price

os.environ["PYTHONIOENCODING"] = "utf-8"
#prepocess function that return request into the training forma
preprocessor = Preprocessor()
Pricer = modal.Cls.from_name("pricer-service", "Pricer")
pricer = Pricer()


if __name__ == "__main__":
    text = preprocessor.preprocess("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
    print(text)
    reply = pricer.price.remote(text)
    print(reply)