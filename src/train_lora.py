import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from peft import LoraConfig, get_peft_model

from vectorize import get_prepared_data
from model import TextClassifier

def evaluate_model(model, data_loader, loss_fn):
    """Функция для расчета ошибки точности и f1-метрики"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1

def run_lora_experiment(rank_value):
    print(f"\n=== Запуск эксперимента с LoRA (Ранг r = {rank_value}) ===")
    X, y = get_prepared_data()
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1
    
    X_tr, X_te, y_tr, y_te = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    base_model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    
    _, base_acc, base_f1 = evaluate_model(base_model, test_loader, loss_fn)
    print(f"""[Baseline] Качество сырой модели до обучения -> Accuracy: {base_acc:.4f}, F1: {base_f1:.4f}""")
    lora_config = LoraConfig(
        r=rank_value,
        lora_alpha=16,
        target_modules=["0"],        
        lora_dropout=0.1,
        #task_type="SEQ_CLS"
    )
    
    model = get_peft_model(base_model, lora_config)
    
    print("Проверка обучаемых параметров: ")
    model.print_trainable_parameters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    num_epochs = 2
    
    print(f"Старт короткого дообучения адаптера на {num_epochs} эпохи...")
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    _, post_acc, post_f1 = evaluate_model(model, test_loader, loss_fn)
    print(f"""[После LoRA r={rank_value}] Метрики -> Accuracy: {post_acc:.4f}, F1: {post_f1:.4f}""")
    
    if rank_value == 8:
        output_dir = "models/lora-adapter"
        model.save_pretrained(output_dir)
        print(f"Веса LoRA-аддаптеров успешно сохранены в папку: {output_dir}")
        
    return post_acc

def main():
    acc_r4 = run_lora_experiment(rank_value=4)
    acc_r16 = run_lora_experiment(rank_value=16)
    
    run_lora_experiment(rank_value=8)
    
    print("\n === Эксперимент успешно завершён ===")
    print(f"""Итоговое сравнение точности: Ранг r=4 -> Accuracy: {acc_r4:.4f} | Ранг r=16 -> Accuracy: {acc_r16:.4f}""")
    
if __name__ == "__main__":
    main()