import os
import pandas as pd 
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / ".." / "data" / "processed" / "clean.csv"
DB_DIR = BASE_DIR / ".." / ".." / "data" / "vector_db"
COLLECTION_NAME = "lenta_clean_index"

def chunk_text_by_tokens(text: str, token_size: int = 200, token_overlap: int = 30) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    
    #исп. стандарт. кодировщик BPE
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + token_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += (token_size - token_overlap)
        
    return chunks

def main():
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")
    
    df = pd.read_csv(DATA_PATH)
    
    print("Инициализация модели эмбеддингов (ruber-base-cased-sentence)...")
    embedder = SentenceTransformer("DeepPavlov/ruber-base-cased-sentence")
    
    print("Настройка локальной ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    print("Обработка текстов и создание чанков...")
    for idx, row in df.iterrows():
        text = str(row['text'])
        topic = str(row['topic']) if 'topic' in df.columns else "unknown"
        
        chunks = chunk_text_by_tokens(text, token_size=200, token_overlap=30)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas({"topic": topic, "doc_id": int(idx)})
            all_ids.append(f"id_{idx}_{i}")
            
    print(f"Всего создано чанков: {len(all_chunks)}. Начинаем генерацию эмбеддингов...")
    vectors = embedder.encode(all_chunks, show_progress_bar=True).tolist()
    
    print("Запись векторов и тексто в ChromaDB...")
    collection.add(
        embeddings=vectors,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"Векторная база сохранена по пути '{DB_DIR}'. Всего записей: {collection.count()}")
    
if __name__ == "__main__":
    main()