import json
from llm_ckient import ask_gigachat

def analyze_review_prompt(review_text: str) -> str:
    
    promt = f"" #через ии сделать промт для гигачата для своего датасета, фото примера есть
    return promt

if __name__ == "__main__":
    user_review = "" #вставить сырой пример для анализа, фото пример, реальное
    
    print("1. Формируем сложный промпт...")
    final_prompt = analyze_review_prompt(user_review)
    
    print("2. Отправляем запрос в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1) #НИЗКАЯ ТЕМПЕРАТУРА ДЛЯ СОБЛЮДЕНИЯ 
    
    print(f"Сырой ответ от модели: \n{raw_response}\n")
    
    print("3. Проверяем валидность полученного json...")
    try:
        parsed_json = json.loads(raw_response.strip())
        print("Успех! Данные успешно преобразованы в Python dict:")
        print(f"Тональность: {parsed_json.get('sentiment')}")
        print(f"Плюсы: {parsed_json.get('pros')}")
        print(f"Минусы: {parsed_json.get('cons')}")
    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула неверный json.")