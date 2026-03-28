import os
import locale
from dotenv import load_dotenv
load_dotenv(override=True)
from support.hello import app, hello

os.environ["PYTHONIOENCODING"] = "utf-8"

#testing if the modal is set up correctly using hello.py
with app.run():
    reply=hello.local()
with app.run():
    reply_remote = hello.remote()



if __name__ == "__main__":
    print(locale.getpreferredencoding()) 
    print(reply)
    print(reply_remote)