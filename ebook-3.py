# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 08:39:53 2025
@author: deepj
"""

# pip install gradio transformers datasets torch watchdog sentencepiece accelerate peft pdfplumber ebooklib beautifulsoup4 faiss-cpu sentence-transformers

import os
import pdfplumber
from ebooklib import epub
from bs4 import BeautifulSoup
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr
import threading

# ===== Configuration =====
MODEL_NAME = "facebook/mbart-large-50"   # multilingual model
MODEL_SAVE_PATH = "ebook_brain_lora_multilingual"
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
MAX_LENGTH = 512
BATCH_SIZE = 1  # reduce due to large model
EPOCHS = 1
LEARNING_RATE = 1e-4
CHUNK_SIZE = 500

# ===== Load Tokenizer & Model =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# LoRA for incremental training
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # for mBART attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print("Multilingual model ready with LoRA.")

# ===== Embedding Model & FAISS =====
embedder = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = None
ebook_chunks = []

# ===== Text Extraction =====
def extract_pdf_text(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_epub_text(file_path):
    text = ""
    try:
        book = epub.read_epub(file_path)
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
    except Exception as e:
        print(f"Error reading EPUB {file_path}: {e}")
    return text

def extract_txt_text(file_path):
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
    return text

def load_ebooks_from_files(file_paths):
    texts = []
    for path in file_paths:
        if path.endswith(".pdf"):
            texts.append(extract_pdf_text(path))
        elif path.endswith(".epub"):
            texts.append(extract_epub_text(path))
        elif path.endswith(".txt"):
            texts.append(extract_txt_text(path))
    return texts

# ===== Tokenization & Training =====
def tokenize_texts(texts):
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        ),
        batched=True,
        remove_columns=["text"]
    )
    return tokenized

def train_model(tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        overwrite_output_dir=False,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=500,
        save_total_limit=5,
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        fp16=False  # Set True if GPU supports it
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model updated!")

# ===== Summarization =====
def summarize_text(text):
    generator = pipeline("text-generation", model=MODEL_SAVE_PATH, tokenizer=tokenizer)
    prompt = f"Summarize the following text concisely in its original language:\n{text}\nSummary:"
    summary = generator(prompt, max_length=150, do_sample=False)[0]['generated_text']
    return summary

# ===== Build FAISS =====
def build_faiss_index(texts):
    global faiss_index, ebook_chunks
    ebook_chunks = []
    embeddings = []

    for text in texts:
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i+CHUNK_SIZE]
            summary = summarize_text(chunk)
            ebook_chunks.append(summary)
            embeddings.append(embedder.encode(summary))

    embeddings = np.array(embeddings).astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    print("FAISS index with multilingual summaries ready.")

# ===== Q&A =====
def ask_question_gui(question):
    if faiss_index is None:
        return "No books loaded yet."
    
    q_emb = embedder.encode(question).astype("float32")
    distances, indices = faiss_index.search(np.array([q_emb]), 3)
    context = " ".join([ebook_chunks[i] for i in indices[0]])

    generator = pipeline("text-generation", model=MODEL_SAVE_PATH, tokenizer=tokenizer)
    prompt = f"Answer the question based on the context below (answer in the same language as question):\nContext: {context}\nQuestion: {question}\nAnswer:"
    answer = generator(prompt, max_length=200, do_sample=True)
    return answer[0]["generated_text"]

# ===== Add New Books =====
def add_new_books(files):
    saved_paths = []
    if not os.path.exists("ebooks"):
        os.makedirs("ebooks")
    for f in files:
        path = os.path.join("ebooks", os.path.basename(f.name))
        with open(path, "wb") as out_file:
            out_file.write(f.read())
        saved_paths.append(path)
    print("New books added:", [os.path.basename(p) for p in saved_paths])
    
    def train_update():
        texts = load_ebooks_from_files(saved_paths)
        tokenized = tokenize_texts(texts)
        train_model(tokenized)
        all_texts = load_ebooks_from_files([os.path.join("ebooks", f) for f in os.listdir("ebooks")])
        build_faiss_index(all_texts)
    
    threading.Thread(target=train_update, daemon=True).start()
    return "Books added and training started with multilingual summarization!"

# ===== GUI =====
iface = gr.Blocks()

with iface:
    gr.Markdown("## 🌏 Multilingual Live eBook Brain Chatbot")
    question_box = gr.Textbox(label="Ask a question (Hindi, English, other languages)")
    answer_box = gr.Textbox(label="Answer")
    submit_btn = gr.Button("Ask")
    upload = gr.File(label="Drag & Drop eBooks (.txt, .pdf, .epub)", file_types=['.txt','.pdf','.epub'], file_types_allow_multiple=True)
    status_box = gr.Textbox(label="Status")

    submit_btn.click(fn=ask_question_gui, inputs=question_box, outputs=answer_box)
    upload.upload(fn=add_new_books, inputs=upload, outputs=status_box)

iface.launch()
