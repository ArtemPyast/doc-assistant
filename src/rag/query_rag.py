import os
import pickle
from pathlib import Path
import faiss
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = (BASE_DIR / ".." / ".." / "data" / "vector_db").resolve()
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

def load_rag_assets():
    """Загружает FAISS индекс и текстовую базу в память один раз при старте сервера."""
    if not FAISS_INDEX_PATH.exists() or not DOCS_PICKLE_PATH.exists():
        raise FileNotFoundError("Ошибка: Файлы базы FAISS не найдены. Сначала запустите build_index.py")
    
    # Легковесный SentenceTransformer загружаем прямо здесь
    from sentence_transformers import SentenceTransformer
    print("Загрузка индекса FAISS и модели эмбеддингов...")
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCS_PICKLE_PATH, "rb") as f:
        db_data = pickle.load(f)
    embedder = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
    
    # Возвращаем словарь со всеми тяжелыми объектами
    return {"index": index, "db_data": db_data, "embedder": embedder}

def rag_answer(question: str, state: dict, k: int = 3):
    """Ищет контекст в FAISS и генерирует ответ через API Ollama."""
    # 1. Получаем эмбеддинг вопроса и ищем в FAISS
    query_vector = state["embedder"].encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    scores, indices = state["index"].search(query_vector, k)
    
    # 2. Собираем найденный контекст и метаданные
    search_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        search_results.append({
            "text": state["db_data"]["chunks"][idx],
            "distance": float(score),
            "meta": state["db_data"]["metadatas"][idx]
        })
    
    context = "\n---\n".join([res["text"] for res in search_results])
    
    # 3. Формируем строгий системный промпт под требования лабораторной
    system_prompt = (
        "Ты - строгий робот-помощник. Твоя задача - ответить на вопрос, основываясь ТОЛЬКО на предоставленных фрагментах текста.\n"
        "Если в предоставленном тексте нет точного ответа на вопрос, ты обязан ответить строго фразой: 'В документах не найдено'.\n"
        "Запрещено использовать внешние знания, придумывать факты или дописывать информацию, которой нет в тексте."
    )
    user_prompt = f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\nВОПРОС: {question}\nОТВЕТ:"
    
    # 4. Отправляем обычный POST-запрос в Ollama (совместимый с OpenAI формат)
    ollama_url = "http://127.0.0.1:11434/v1/chat/completions"
    payload = {
        "model": "qwen2.5:3b",
        "temperature": 0.1,  # Низкая температура для строгости ответов
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    
    llm_response = response.json()["choices"][0]["message"]["content"].strip()
    return llm_response, search_results
