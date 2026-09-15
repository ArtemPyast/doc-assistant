import torch
import torch.nn as nn
from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("Текстовый старт")
    
    X, y = get_prepared_data()
    
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1
    
    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nСтркуктура нейросети:")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    logits = model(x)
    
    loss = loss_fn(logits, y)
    print("\n[Замер] Значение ошибки (Loss) ДО шага обучения:", loss.item())
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    new_logits = model(X)
    new_loss=loss_fn(new_logits,y)
    print("\n[Замер] Значение ошибки (Loss) ПОСЛЕ шага обучения:", new_loss.item())
    
    if new_loss.item() < loss.item():
        print("Веса скорректированный, ошибка модели пошла вниз")
        
if __name__ == "__main__":
    main()