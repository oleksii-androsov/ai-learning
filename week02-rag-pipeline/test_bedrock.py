import boto3
import json
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-runtime", region_name="eu-central-1")

response = client.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({
        "inputText": "Hello, this is a test sentence.",
        "dimensions": 512,
        "normalize": True
    })
)

result = json.loads(response["body"].read())

print(f"Embedding length: {len(result['embedding'])}")
print(f"First 5 values: {result['embedding'][:5]}")
