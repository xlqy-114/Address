import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from tqdm import tqdm

# ------------------ 加载并合并数据 ------------------
df_human = pd.read_excel('humancheck.xlsx')
df_human['weight'] = 1.0

df_conf = pd.read_excel('confidence1.xlsx')
df_conf['weight'] = 0.2

df_all = pd.concat([df_human, df_conf], ignore_index=True)

# ------------------ 标签编码 ------------------
le = LabelEncoder()
df_all['分类结果'] = le.fit_transform(df_all['分类结果'])

# ------------------ 文本编码 ------------------
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
encoded_all = tokenizer.batch_encode_plus(
    df_all['地址'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

input_ids = encoded_all['input_ids']
attention_mask = encoded_all['attention_mask']
labels = torch.tensor(df_all['分类结果'].values)
weights = torch.tensor(df_all['weight'].values, dtype=torch.float32)

# ------------------ 数据集划分 ------------------
train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, \
train_labels, val_labels, train_weights, val_weights = train_test_split(
    input_ids, attention_mask, labels, weights, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_weights)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels, val_weights)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# ------------------ 模型定义 ------------------
model = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=len(le.classes_))
optimizer = AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True, factor=0.1)

# ------------------ 训练函数 ------------------
def train(model, train_dataloader, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_ids, attention_mask, labels, weights = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            weighted_loss = (loss * weights).mean()

        scaler.scale(weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += weighted_loss.item()
    return total_loss / len(train_dataloader)

# ------------------ 验证函数 ------------------
def evaluate(model, val_dataloader):
    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(val_dataloader):
        input_ids, attention_mask, labels, _ = [x.to(device) for x in batch]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=le.classes_)
    return accuracy, report

# ------------------ 模型训练 ------------------
best_val_accuracy = 0.0
epochs = 10
patience = 2
patience_counter = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = train(model, train_dataloader, optimizer, scaler)
    print(f"训练损失: {train_loss:.4f}")

    accuracy, report = evaluate(model, val_dataloader)
    print(f"验证准确率: {accuracy:.4f}")
    print(report)

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        model.save_pretrained('best_model')
        patience_counter = 0
    else:
        patience_counter += 1

    scheduler.step(accuracy)

    if patience_counter >= patience:
        print("验证准确率没有提升，提前停止训练。")
        break

# ------------------ 保存最终模型 ------------------
model.save_pretrained('fine_tuned_address_model')
print("模型训练完成，最终模型已保存为 'fine_tuned_address_model'。")