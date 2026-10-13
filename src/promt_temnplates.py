import json
from src.llm_client import ask_gigachat


def analyze_news_prompt(news_title: str, news_text: str) -> str:
    """
    Конструирует структурированный промпт с Few-shot примерами для анализа новостей на фейк.
    """
    prompt = f"""
Ты — профессиональный ИИ-аналитик по проверке фактов (Fact-Checker). Твоя задача — проанализировать новостную статью и определить, является ли она фейковой (Fake) или достоверной (Real).

Выведи результат СТРОГО в формате JSON. Не пиши никаких вводных слов, markdown-разметки или пояснений вне JSON.

### ПРИМЕРЫ:

Входная новость:
Заголовок: "BREAKING: Pope Francis Just Called Out Donald Trump During His Christmas Speech"
Текст: "Pope Francis used his annual Christmas Day message to rebuke Donald Trump without even mentioning his name. The Pope delivered his message just days after members of the United Nations condemned Trump's move to recognize Jerusalem as the capital of Israel..."
Выходной JSON:
{{"label": "Real", "reason": "Новость ссылается на реальное событие (рождественское обращение Папы Римского) и содержит конкретные политические детали, соответствующие новостной повестке. Тон нейтральный, без явных признаков сенсационности или пропаганды."}}

Входная новость:
Заголовок: "Trump Is So Obsessed He Even Has Obama’s Name Coded Into His Website (IMAGES)"
Текст: "On Christmas day, Donald Trump announced that he would 'be back to work' the following day, but he is golfing for the fourth day in a row. The former reality show star blasted former President Barack Obama for playing golf and now Trump is on track to outpace the number of golf games his predecessor played..."
Выходной JSON:
{{"label": "Fake", "reason": "Заголовок использует сенсационное и эмоционально окрашенное слово 'Obsessed'. Текст содержит субъективные оценки ('former reality show star', 'blasted') и не подтвержденные факты, направленные на создание негативного образа. Это характерно для предвзятой или фейковой новости."}}

### РЕАЛЬНОЕ ЗАДАНИЕ:
Входная новость:
Заголовок: "{news_title}"
Текст: "{news_text}"
Выходной JSON:
"""
    return prompt


if __name__ == "__main__":
    # --- СЫРЫЕ ДАННЫЕ ДЛЯ АНАЛИЗА (Взято из твоего Fake.csv) ---
    # Пример №1 (вторая новость из файла)
    raw_title = "Drunk Bragging Trump Staffer Started Russian Collusion Investigation"
    raw_text = "House Intelligence Committee Chairman Devin Nunes is going to have a bad day. He s been under the assumption, like many of us, that the Christopher Steele-dossier was what prompted the Russia investigation so he s been lashing out at the Department of Justice and the FBI in order to protect Trump. As it happens, the dossier is not what started the investigation, according to documents obtained by the New York Times. Former Trump campaign adviser George Papadopoulos was drunk in a wine bar when he revealed knowledge of Russian opposition research on Hillary Clinton..."

    print("1. Формируем сложный промпт...")
    # Передаем ДВА аргумента, как и требует функция
    final_prompt = analyze_news_prompt(raw_title, raw_text)

    print("2. Отправляем запрос в GigaChat...")
    # Отправляем промпт в нейросеть
    raw_response = ask_gigachat(final_prompt, temperature=0.1)

    print(f"\nСырой ответ от модели: \n{raw_response}\n")

    print("3. Проверяем валидность полученного JSON...")
    try:
        # Очищаем ответ от возможных markdown-тегов (```json ... ```)
        clean_response = raw_response.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(clean_response)

        print("Успех! Данные успешно преобразованы в Python dict:")
        print(f"Вердикт: {parsed_json.get('label')}")
        print(f"Причина: {parsed_json.get('reason')}")

    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула неверный JSON.")