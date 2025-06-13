import sqlite3

# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Function to insert a message
def save_message(sender, message):
    """Save a chat message to the database."""
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

# Function to retrieve chat history
def get_chat_history(limit=10):
    """Retrieve the last few chat messages."""
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT sender, message, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
    history = cursor.fetchall()
    
    conn.close()
    return history

# Example Usage
def chat_with_ai():
    """Simple chatbot loop that stores user & AI messages in history."""
    import random  # Replace with an actual AI model later
    
    responses = ["Hello!", "How can I help?", "Tell me more!", "That's interesting!", "Can you elaborate?"]

    print("Chatbot started! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Save user's message
        save_message("User", user_input)
        
        # Generate AI response
        ai_response = random.choice(responses)  # Replace with an AI model
        print(f"AI: {ai_response}")
        
        # Save AI's response
        save_message("AI", ai_response)

    print("\nChat history:")
    for sender, message, timestamp in reversed(get_chat_history()):
        print(f"[{timestamp}] {sender}: {message}")

if __name__ == "__main__":
    chat_with_ai()

get_chat_history()