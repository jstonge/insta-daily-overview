import os

# Import the SDK and the client module
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv
import httpx

load_dotenv()

custom_httpx_client = httpx.Client(verify=False)

# Connect to the Label Studio API 
client = LabelStudio(
    base_url=os.getenv('LABEL_STUDIO_URL'), 
    api_key=os.getenv('LABEL_STUDIO_API_KEY'),
    httpx_client=custom_httpx_client
)

# We created the project manually on the platform
PROJECT_ID = 98


if __name__ == "__main__":
    # A basic request to verify connection is working
    me = client.users.whoami()
    print("username:", me.username)
    print("email:", me.email)