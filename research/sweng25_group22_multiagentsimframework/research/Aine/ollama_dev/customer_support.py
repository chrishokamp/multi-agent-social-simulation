import sqlite3
from ollama import chat

# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

def streaming_chat(user_input):
    try:
        print("\nAI: ", end="", flush=True)
        stream = chat(model='llama3.2', messages=[{'role': 'user', 'content': user_input}], stream=True)

        ai_response = ""  # Store AI response
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            ai_response += content  # Append AI response
        
        print()  
        save_message("AI", ai_response)  # Save AI response

    except Exception as e:
        print(f"\nError: {str(e)}")

def save_message(sender, message):
    """Save a chat message to the database, except exit/quit."""
    if message.lower() in {"exit", "quit"}:
        return  # Do not save exit commands

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)  # Ensure table exists

    cursor.execute("INSERT INTO chat_history (sender, message) VALUES (?, ?)", (sender, message))
    conn.commit()
    conn.close()

def get_chat_history(limit=10):
    """Retrieve the last few chat messages."""
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT sender, message, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
    history = cursor.fetchall()
    
    conn.close()
    return history

def main():
    print("Welcome to Customer Support! How may I help you?")
    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break  # Exit without saving

        save_message("User", user_input)  # Save only valid user input
        streaming_chat(user_input)

    print("\nChat history:")
    for sender, message, timestamp in reversed(get_chat_history()):
        print(f"[{timestamp}] {sender}: {message}")

if __name__ == "__main__":
    main()
