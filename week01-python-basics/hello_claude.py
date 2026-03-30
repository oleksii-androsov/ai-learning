import anthropic
import os
from dotenv import load_dotenv


def load_document(filepath):
    if not os.path.exists(filepath):
        print(f"Error: file '{filepath}' not found.")
        return None
    with open(filepath, "r") as f:
        return f.read()


def chat(client, conversation_history, system_prompt, user_input):
    conversation_history.append({"role": "user", "content": user_input})

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history
        )
        reply = message.content[0].text
        conversation_history.append({"role": "assistant", "content": reply})
        return reply

    except anthropic.APIConnectionError:
        print("Error: could not connect to the API. Check your internet connection.")
        return None
    except anthropic.AuthenticationError:
        print("Error: invalid API key. Check your .env file.")
        return None
    except anthropic.APIStatusError as e:
        print(f"API error: {e.message}")
        return None


def main():
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found. Check your .env file.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    document = load_document("week01-python-basics/sample.txt")
    if not document:
        return

    system_prompt = f"You are a helpful assistant. Use this document to answer questions:\n\n{document}"

    print("Chat with Claude about the document. Type 'quit' to exit.")

    conversation_history = []

    while True:
        user_input = input("\nYou: ")

        if user_input == "quit":
            print("Goodbye!")
            break

        reply = chat(client, conversation_history, system_prompt, user_input)
        if reply is None:
            break
        print(f"\nClaude: {reply}")


main()
