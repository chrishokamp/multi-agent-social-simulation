import sqlite3

# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Function to insert a message
def save_message(sender, message):
    cursor.execute("INSERT INTO chat_history (sender, message) VALUES (?, ?)", (sender, message))
    conn.commit()

# Function to retrieve chat history
def get_chat_history(limit=10):
    cursor.execute("SELECT sender, message, timestamp FROM chat_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    return cursor.fetchall()

# Example Usage
save_message("Aine", "How's it going?")
save_message("Alex", "Not much, just enjoying myself!")
save_message("Aine", "Im glad!")

# Fetch and print chat history
history = get_chat_history()
for row in history:
    print(row)


# Close the connection
conn.close()
