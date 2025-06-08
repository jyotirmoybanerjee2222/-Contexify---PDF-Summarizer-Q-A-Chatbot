# app.py
import streamlit as st
import os
import fitz  # PyMuPDF
from summarizer import summarize_text
from auth import login, signup, is_valid_email
from qna import get_pdf_text, get_text_chunks, create_vector_store, ask_question

def login_page():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        success, message = login(email, password)
        if success:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.success(f"Welcome {email}!")
        else:
            st.error(message)

def signup_page():
    st.title("📝 Sign Up")
    username = st.text_input("Username")
    email = st.text_input("Email Address")
    password = st.text_input("Choose Password", type="password")

    if st.button("Sign Up"):
        if not username or not email or not password:
            st.error("All fields are required.")
        elif not is_valid_email(email):
            st.error("Invalid email format.")
        else:
            success, message = signup(username, email, password)
            if success:
                st.success(message)
            else:
                st.error(message)

def app_page():
    st.title("📄 Contexify - PDF Summarizer & Q&A Chatbot")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

    if pdf_docs:
        if st.button("Summarize PDFs"):
            full_text = get_pdf_text(pdf_docs)
            summary = summarize_text(full_text)
            st.subheader("📝 Summary")
            st.write(summary)

        if st.button("Index for Q&A"):
            with st.spinner("Processing and indexing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("PDFs indexed successfully!")

    user_question = st.text_input("Ask a question from the uploaded PDF")
    if user_question:
        response = ask_question(user_question)
        st.write("💬 Reply:", response)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.success("Logged out successfully.")

def main():
    st.set_page_config("Contexify PDF Tool")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    menu = st.sidebar.selectbox(
        "Menu", ["Login", "Sign Up"] if not st.session_state.logged_in else ["Home", "Logout"]
    )

    if not st.session_state.logged_in:
        if menu == "Login":
            login_page()
        elif menu == "Sign Up":
            signup_page()
    else:
        if menu == "Home":
            app_page()
        elif menu == "Logout":
            st.session_state.logged_in = False
            st.success("Logged out successfully.")

if __name__ == "__main__":
    main()
