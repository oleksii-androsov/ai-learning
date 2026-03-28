import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

print("Chat with Claude. Type 'quit' to exit.")

while True:
    user_input = input("\nYour question: ")

    if user_input == "quit":
        print("Goodbye!")
        break

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print(message.content[0].text)
