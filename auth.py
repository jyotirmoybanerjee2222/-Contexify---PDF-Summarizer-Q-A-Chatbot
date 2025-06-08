# auth.py

import re
import requests

import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
import os


cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

load_dotenv()
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")

FIREBASE_AUTH_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def extract_firebase_error(response):
    try:
        error = response.json()['error']['message']
        return error.replace('_', ' ').capitalize()
    except:
        return "An unknown error occurred."

def signup(username, email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    url = f"{FIREBASE_AUTH_BASE}:signUp?key={FIREBASE_API_KEY}"

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        user_data = response.json()
        auth.update_user(user_data['localId'], display_name=username)
        return True, "User created successfully."
    except requests.exceptions.HTTPError as e:
        return False, extract_firebase_error(response)

def login(email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    url = f"{FIREBASE_AUTH_BASE}:signInWithPassword?key={FIREBASE_API_KEY}"

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True, email
    except requests.exceptions.HTTPError as e:
        return False, extract_firebase_error(response)
