import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump({}, f)
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(email, data):
    history = load_history()
    history.setdefault(email, [])
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    history[email].append(data)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_user_history(email):
    history = load_history()
    return history.get(email, [])

def clear_user_history(email):
    """Clear all history for a specific user"""
    try:
        history = load_history()
        
        if email in history:
            # Clear the history for this user
            history[email] = []
            
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)
            
            return True
        else:
            # User has no history, but that's fine
            return True
            
    except Exception as e:
        print(f"Error clearing history for {email}: {e}")
        return False