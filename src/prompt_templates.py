import json
from llm_client import ask_gigachat

def analyze_review_prompt(review_text: str) -> str:
    prompt = f"""
"""
    return prompt

if __name__ == "__main__":
    user_review = "отзыв"
    
print("Формируем сложный промпт")
final_promt = analyze_review_prompt(user_review)
print("Отправляем запрос в gigachat")
raw_response = ask_gigachat(final_prompt, temperature=0.1)
print(f"Сырой ответ от модели:\n{raw_responce}\n")

print("Проверяем валидность поулченного JSON")
try:
    parsed_json = json.loads(raw_response.strip())
    print("Успех")
    print(f"Тональность:  {parsed_json.get('sentiment')}")
    print(f"Плюсы:  {parsed_json.get('pros')}")
    print(f"Минусы:  {parsed_json.get('cons')}")
except json.JSONDecodeError:
    print("Ошибка: модель нарушила формат")