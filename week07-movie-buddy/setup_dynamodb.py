"""Run once to create the three DynamoDB tables for Movie Buddy memory."""
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("dynamodb", region_name="eu-central-1")

TABLES = [
    {
        "TableName": "movie_buddy_profiles",
        "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "user_id", "AttributeType": "S"}],
        "BillingMode": "PAY_PER_REQUEST",
    },
    {
        "TableName": "movie_buddy_summaries",
        "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "user_id", "AttributeType": "S"}],
        "BillingMode": "PAY_PER_REQUEST",
    },
    {
        "TableName": "movie_buddy_devices",
        "KeySchema": [{"AttributeName": "device_token", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "device_token", "AttributeType": "S"}],
        "BillingMode": "PAY_PER_REQUEST",
    },
]

for table in TABLES:
    try:
        client.create_table(**table)
        print(f"Created table: {table['TableName']}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table already exists: {table['TableName']}")
        else:
            raise

print("Done.")
