import os
import pickle
import pandas as pd
from pathlib import Path
import faiss #pip install faiss-cpu
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import nltk

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / ".." / "data" / "processed" / "clean.csv"
DB_DIR = BASE_DIR / ".." / ".." / "data" / "vector_db"

# Пути к файлам новой базы FAISS
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def chunk_text_by_sentences(text: str, max_tokens: int = 300, overlap_sentences_count: int = 1) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = nltk.sent_tokenize(text, language="russian")
    chunks, current_chunk, current_tokens_count = [], [], 0

    for sentence in sentences:
        sentence_tokens_count = len(tokenizer.encode(sentence, add_special_tokens=False))
        if sentence_tokens_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_tokens_count = [], 0
            words = sentence.split()
            sub_chunk, sub_tokens = [], 0
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if sub_tokens + word_tokens > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk, sub_tokens = [word], word_tokens
                else:
                    sub_chunk.append(word)
                    sub_tokens += word_tokens
            if sub_chunk:
                current_chunk, current_tokens_count = sub_chunk, sub_tokens
            continue

        if current_tokens_count + sentence_tokens_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            if overlap_sentences_count > 0 and len(current_chunk) >= overlap_sentences_count:
                current_chunk = current_chunk[-overlap_sentences_count:]
                current_tokens_count = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
            else:
                current_chunk, current_tokens_count = [], 0

        current_chunk.append(sentence)
        current_tokens_count += sentence_tokens_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")
        
    df = pd.read_csv(DATA_PATH)
    
    print("Загрузка локальной модели эмбеддингов...")
    model = SentenceTransformer(MODEL_NAME)
    
    all_chunks = []
    all_metadatas = []
    
    print("Разбиение текстов на чанки...")
    for row in df.itertuples(index=True):
        text = str(row.text)
        topic = str(row.topic) if hasattr(row, 'topic') else "unknown"
        doc_id = row.Index
        
        chunks = chunk_text_by_sentences(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"topic": topic, "doc_id": int(doc_id)})
            
    print(f"Всего чанков подготовлено к векторизации: {len(all_chunks)}")
    
    print("Генерация эмбеддингов через SentenceTransformer (это может занять время)...")
    # Преобразуем тексты в векторы. Нормализуем их для косинусного сходства
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings) 
    
    print("Создание и наполнение индекса FAISS...")
    dimension = embeddings.shape[1]
    # IndexFlatIP + нормализованные векторы = Косинусное сходство
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Используем pathlib для явного создания папки
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss_index_str = str(FAISS_INDEX_PATH.resolve())
    docs_pickle_str = str(DOCS_PICKLE_PATH.resolve())
    
    print(f"Сохранение векторного индекса в: {faiss_index_str}")
    faiss.write_index(index, faiss_index_str)
    
    print(f"Сохранение текстовых соответствий в: {docs_pickle_str}")
    with open(docs_pickle_str, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadatas": all_metadatas}, f)

    print(f"Успешно! База FAISS создана. Всего векторов в индексе: {index.ntotal}")

if __name__ == "__main__":
    main()
