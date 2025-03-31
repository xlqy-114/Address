import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------- 加载数据 --------------
file_path = 'Train.xlsx'
df = pd.read_excel(file_path)

# -------------- 数据预处理 --------------
le = LabelEncoder()
df['分类结果'] = le.fit_transform(df['分类结果'])

# -------------- 文本数据预处理 --------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, max_length=128):
    return tokenizer.batch_encode_plus(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
    )

# 地址编码
encoded_data = encode_texts(df['地址'].tolist())

# -------------- 创建训练集和验证集 --------------
y = df['分类结果'].values
train_texts, val_texts, train_labels, val_labels = train_test_split(df['地址'], y, test_size=0.2, random_state=42)

# 对训练集和验证集的地址进行编码
train_encoded = encode_texts(train_texts.tolist())
val_encoded = encode_texts(val_texts.tolist())

# 获取 input_ids 和 attention_mask
train_input_ids = train_encoded['input_ids']
val_input_ids = val_encoded['input_ids']
train_attention_mask = train_encoded['attention_mask']
val_attention_mask = val_encoded['attention_mask']

# 确保形状一致
assert train_input_ids.shape[0] == len(train_labels), f"训练数据大小不匹配: {train_input_ids.shape[0]} != {len(train_labels)}"
assert val_input_ids.shape[0] == len(val_labels), f"验证数据大小不匹配: {val_input_ids.shape[0]} != {len(val_labels)}"

# 转换标签为 Tensor
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

print(f"训练集大小: {train_input_ids.shape[0]}, 验证集大小: {val_input_ids.shape[0]}")

# -------------- 创建PyTorch数据集 --------------
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)  
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# -------------- 定义模型 --------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_)) 

optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 使用GPU训练
model.to(device)        

# -------------- 训练模型 --------------
scaler = GradScaler()  # 混合精度训练

# 使用学习率衰减策略
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True, factor=0.1)

def train(model, train_dataloader, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        
        optimizer.zero_grad()

        with autocast():  # 开启自动混合精度
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def evaluate(model, val_dataloader):
    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(val_dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=le.classes_)
    return accuracy, report

# -------------- 训练 -----------------
best_val_accuracy = 0.0 # 最佳验证准确率
epochs = 10  # 10轮训练
patience = 2  # 早停条件
patience_counter = 0    # 早停计数器

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 训练
    train_loss = train(model, train_dataloader, optimizer, scaler)
    print(f"训练损失: {train_loss}")
    
    # 验证
    accuracy, report = evaluate(model, val_dataloader)
    print(f"验证准确率: {accuracy}")
    print(report)

    # 如果验证准确率提升，则重置早停计数器并保存最佳模型
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        model.save_pretrained('best_model')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 使用学习率调度器
    scheduler.step(accuracy)

    # 如果验证准确率在连续的 patience 个 epoch 内没有提升，则提前停止
    if patience_counter >= patience:
        print("验证准确率没有提升，提前停止训练。")
        break

# -------------- 保存最佳模型 --------------
model.save_pretrained('fine_tuned_bert_model2')
