import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag.query_rag import load_rag_assets, rag_answer

# Сюда запишем тяжелые ресурсы при старте
index_state = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_state
    print("=== Загрузка RAG-системы ===")
    index_state = load_rag_assets()  # Строим/загружаем индекс один раз при старте сервера
    print("=== Система успешно загружена и готова к запросам ===")
    yield
    print("=== Остановка сервера ===")

app = FastAPI(title="Doc Assistant", lifespan=lifespan)

# Pydantic-схема: описывает форму запроса, FastAPI сам её валидирует и документирует
class Query(BaseModel):
    question: str = Field(..., description="Вопрос к базе документов")
    k: int = Field(default=3, ge=1, le=10, description="Количество фрагментов для поиска")

@app.get("/health")
def health():
    """Эндпоинт для проверки здоровья сервиса"""
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Query):
    """Точка входа для RAG вопросов"""
    # Обязательная обвязка для обработки ошибок (пустой вопрос)
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    
    # Замеряем время ответа, чтобы проанализировать производительность CPU
    start_time = time.time()
    
    answer, sources = rag_answer(q.question, index_state, k=q.k)
    
    elapsed_time = time.time() - start_time
    print(f"Запрос обработан за {elapsed_time:.4f} сек.")  # Логирование в консоль
    
    return {
        "answer": answer, 
        "sources": sources,
        "metrics": {"execution_time_seconds": round(elapsed_time, 2)}
    }
