import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

with open("week01-python-basics/sample.txt", "r") as f:
    document = f.read()

system_prompt = f"You are a helpful assistant. Use this document to answer questions:\n\n{document}"

print("Chat with Claude about the document. Type 'quit' to exit.")

conversation_history = []

while True:
    user_input = input("\nYou: ")

    if user_input == "quit":
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_input})

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history
    )

    reply = message.content[0].text

    conversation_history.append({"role": "assistant", "content": reply})

    print(f"\nClaude: {reply}")
