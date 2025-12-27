import json
import os
import hashlib

USER_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# ---------- UTIL ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ---------- SIGN UP ----------
def signup(email, password):
    users = load_users()

    if email in users:
        return False, "Email already registered"

    users[email] = {
        "password": hash_password(password)
    }
    save_users(users)
    return True, "Signup successful"

# ---------- LOGIN ----------
def login(email, password):
    users = load_users()
    if email not in users:
        return False

    return users[email]["password"] == hash_password(password)
