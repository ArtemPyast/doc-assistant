import os
import pickle
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
from gigachat import GigaChat
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = (BASE_DIR / ".." / ".." / "data" / "vector_db").resolve()
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

# 1. Локальные компоненты инициализируем при старте
MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"

if not FAISS_INDEX_PATH.exists() or not DOCS_PICKLE_PATH.exists():
    raise FileNotFoundError("Ошибка: Файлы базы FAISS не найдены. Сначала запустите build_index.py")

print("Загрузка векторного индекса FAISS и текстов...")
index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(DOCS_PICKLE_PATH, "rb") as f:
    db_data = pickle.load(f)

# Инициализируем модель для кодирования входящих вопросов
embedder = SentenceTransformer(MODEL_NAME)

# 2. Инициализируем GigaChat
credentials = os.getenv("GIGACHAT_CREDENTIALS")
if not credentials:
    raise ValueError("Ошибка: Переменная GIGACHAT_CREDENTIALS не найдена в .env")

giga_client = GigaChat(credentials=credentials, verify_ssl_certs=False)

def search_in_db(query: str, k: int = 3):
    """Ищет k ближайших векторов в индексе FAISS."""
    # Переводим текст вопроса в вектор и нормализуем
    query_vector = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    # Ищем совпадения. Метод возвращает скоры (близость) и внутренние ID (индексы массива)
    scores, indices = index.search(query_vector, k)
    
    retrieved_data = []
    # Извлекаем данные из плоских массивов результатов FAISS
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # Значение -1 возвращается, если в базе документов меньше, чем k
            continue
            
        retrieved_data.append({
            "text": db_data["chunks"][idx],
            # Для IndexFlatIP чем больше score (ближе к 1.0), тем точнее совпадение
            "distance": float(score), 
            "meta": db_data["metadatas"][idx]
        })
            
    return retrieved_data

def ask_gigachat(prompt: str, temperature: float = 0.3) -> str:
    response = giga_client.chat({
        "model": "GigaChat-3-Lightning",
        "temperature": temperature,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    return response.choices[0].message.content

def rag_answer(question: str, k: int = 3):
    """Объединяет поиск контекста и генерацию ответа с помощью GigaChat."""
    search_results = search_in_db(question, k=k)
    
    context = "\n---\n".join([res["text"] for res in search_results])
    
    prompt = (
        "Ты - строгий робот-помощник. Твоя задача - ответить на вопрос, основываясь ТОЛЬКО на предоставленных фрагментах текста.\n"
        "Если в предоставленном тексте нет точного ответа на вопрос, ты обязан ответить строго фразой: 'В документах не найдено'.\n"
        "Запрещено использовать внешние знания, придумывать факты или дописывать информацию, которой нет в тексте.\n\n"
        f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\n"
        f"ВОПРОС: {question}\n"
        "ОТВЕТ:"
    )
    
    llm_response = ask_gigachat(prompt)
    return llm_response, search_results

if __name__ == "__main__":
    print("RAG-конвейер на FAISS успешно запущен.\n")
    
    fake_question = "Рецепт приготовления пиццы Маргарита."
    valid_question = "Кто тренировал \"Мисс Германии\"?"

    print(f"ТЕСТ 1: Вопрос вне корпуса данных")
    ans, _ = rag_answer(fake_question, k=3)
    print(f"Вопрос: {fake_question}")
    print(f"Ответ модели: {ans}\n")

    print(f"ТЕСТ 2: Сравнение параметра k")
    for current_k in [1, 2, 3]:
        ans, sources = rag_answer(valid_question, k=current_k)
        print(f"--- Результат при k = {current_k} ---")
        print(f"Ответ модели:\n{ans}\n")
        print("Использованные источники:")
        for num, s in enumerate(sources, 1):
            # В FAISS с косинусным сходством близость (Score) отображает релевантность напрямую
            print(f"  {num}. [Score: {s['distance']:.4f}] Текст: {s['text'][:70]}...")
        print("-" * 40)
        
    giga_client.close()
