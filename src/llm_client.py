import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()

def ask_gigachat(prompt: str, temperature: float = 0.3) -> str:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    
    if not credentials:
        raise  ValueError("Ошибка: Переменная GIGACHAT_CREDENTIALS не найдена в .env")
    
    with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
        response = giga.chat({
            "model": "GigaChat",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        return response.choices[0].messages.content
    
if __name__ == "__main__":
    print("Проверка связи c Gigachat")
    test_question = "Что такое промпт-инжиниринг в трех предложениях?"
    try:
        answer = ask_gigachat(test_question)
        print(f"\n вопрос: {test_question}")
        print(f"Ответ: \n{answer}")
    except Exception as e:
        print(f"Произошла ошибка при подключении {e}")