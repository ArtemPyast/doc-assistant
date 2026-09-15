import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def get_prepared_data():
    df = pd.read_csv("data/processed/clean.csv")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_numpy = vectorizer.fit_transform(df["text"]).toarray()
    
    label_encoder = LabelEncoder()
    y_numpy = label_encoder.fit_transform(df["topic"]) #поменять topic на свое, на то что классифицирует
    
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.long)
    
    print("Обнаружение классов (тем): {len(label_encoder.classes_)}")
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    get_prepared_data()
