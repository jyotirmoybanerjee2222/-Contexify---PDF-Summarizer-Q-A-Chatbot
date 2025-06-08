import re
import spacy
import gc
from transformers import pipeline

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model: {e}")

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_spacy(text, max_chunk_len=1024):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_len:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def split_text_technical(text, max_chunk_len=1000, overlap=100):
    sections = re.split(
        r'\n\s*(Abstract|Introduction|Method|Experiment|Result|Discussion|Conclusion|References)\s*\n',
        text,
        flags=re.IGNORECASE
    )

    chunks = []
    for section in sections:
        if not section.strip():
            continue

        doc = nlp(section)
        sentences = [sent.text.strip() for sent in doc.sents]

        start_idx = 0
        while start_idx < len(sentences):
            end_idx = start_idx + max_chunk_len
            chunk = " ".join(sentences[start_idx:end_idx])
            chunks.append(chunk)
            start_idx = end_idx - overlap if end_idx - overlap > 0 else end_idx

    return chunks

def summarize_text(text):
    try:
        cleaned = clean_text(text)
    except ValueError as ve:
        return f"Input Error: {ve}"

    # Determine chunking method
    sections = re.split(
        r'\n\s*(Abstract|Introduction|Method|Experiment|Result|Discussion|Conclusion|References)\s*\n',
        cleaned,
        flags=re.IGNORECASE
    )

    if len(sections) < 2:
        chunks = split_text_spacy(cleaned)
    else:
        chunks = split_text_technical(cleaned)

    if not chunks:
        return "Error: No valid content found after chunking."

    # Stage 1: Primary summarization
    try:
        primary_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        return f"Error loading primary summarizer model: {e}"

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = primary_summarizer(
                chunk,
                max_length=120,
                min_length=60,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.2,
                repetition_penalty=2.0
            )[0]['summary_text']
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Warning: Skipping chunk {i} due to summarization error: {e}")
            continue

    del primary_summarizer
    gc.collect()

    if not chunk_summaries:
        return "Error: Summarization failed for all chunks."

    # Stage 2: Final summarization
    combined_summary = " ".join(chunk_summaries)
    if len(combined_summary) > 4000:
        combined_summary = combined_summary[:4000]

    try:
        final_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        return f"Error loading final summarizer model: {e}"

    try:
        final_summary = final_summarizer(
            combined_summary,
            max_length=150,
            min_length=75,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.1
        )[0]['summary_text']
    except Exception as e:
        final_summary = f"Final summarization failed: {e}"

    del final_summarizer
    gc.collect()

    return final_summary
