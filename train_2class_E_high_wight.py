from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from network17 import MAFNet  # ????
from data_loader import DMRI_CellType_BinaryDataset  # ???????

# --- ?? ---
random_seed = 42
batch_size = 64
num_epochs = 200
learning_rate = 1e-4
weight_decay = 1e-5

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_csv = '/project/AIRC/NWang_lab/shared/rui/Cell_type/separate_E_I/dMRIcelltype_N54943_10timesres_trainingdata_MRIonly_allslices_addEI_nooverlapping_cleaned_E_highweight_training.csv'
test_csv = '/project/AIRC/NWang_lab/shared/rui/Cell_type/separate_E_I/dMRIcelltype_N54943_10timesres_trainingdata_MRIonly_allslices_addEI_nooverlapping_cleaned_E_highweight_testing.csv'

train_dataset = DMRI_CellType_BinaryDataset(train_csv)
test_dataset = DMRI_CellType_BinaryDataset(test_csv)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from collections import Counter
labels = train_dataset.labels
cnt = Counter(labels)
total = len(labels)
print("Train set label count:", cnt)
for label in sorted(cnt):
    print(f"  Label {label} count: {cnt[label]}  ({cnt[label] / total:.2%})")


model = MAFNet(num_classes=2, seq_len=17).to(device)
class_weights = torch.tensor([0.41, 0.59], dtype=torch.float32).to(device)  # ????????
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

save_dir = '/project/AIRC/NWang_lab/shared/rui/Cell_type/separate_E_I/result/results_mafnet_E_highte_weight41_59'
os.makedirs(save_dir, exist_ok=True)

txt_log_path = os.path.join(save_dir, 'epoch_log.txt')
if os.path.exists(txt_log_path):
    os.remove(txt_log_path)

best_acc = 0
train_log = []

target_names = ['others', 'E']  # 0: others, 1: EI

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')

    # --- Train ---
    model.train()
    total_loss = 0
    for x, y, _ in tqdm(train_loader, desc='Train', leave=False):
        x = x.unsqueeze(1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # --- Eval ---
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, desc='Eval', leave=False):
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    test_loss = total_loss / len(test_loader)
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='weighted')

    train_log.append({
        'epoch': epoch+1,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1
    })

    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)

    with open(txt_log_path, 'a') as f:
        f.write(f"===== Epoch {epoch+1} =====\n")
        f.write(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\n\n")

    # ----------- ???? -----------
    plt.figure()
    plt.plot([x['train_loss'] for x in train_log], label='Train Loss')
    plt.plot([x['test_loss'] for x in train_log], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot([x['test_acc'] for x in train_log], label='Test Accuracy')
    plt.plot([x['test_f1'] for x in train_log], label='Test F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_metric_curve.png'))
    plt.close()

    pd.DataFrame(train_log).to_csv(os.path.join(save_dir, 'train_log.csv'), index=False)

    # --------- ????????? ---------
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        print("Best model saved.")

        best_true, best_pred, best_coords = [], [], []
        with torch.no_grad():
            for x, y, coords in tqdm(test_loader, desc='BestModelTestSet', leave=False):
                x = x.unsqueeze(1).to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                best_pred.extend(preds)
                best_true.extend(y.numpy())
                best_coords.append(coords.numpy())
        best_coords = np.concatenate(best_coords, axis=0)
        save_result = pd.DataFrame({
            'true_label': best_true,
            'pred_label': best_pred,
            'z': best_coords[:, 0],
            'x': best_coords[:, 1],
            'y': best_coords[:, 2]
        })
        save_result.to_csv(os.path.join(save_dir, 'prediction_result_best.csv'), index=False)
        print("Best model prediction_result_best.csv saved.")

print("\n===> Final test: save confusion matrix and sample-wise csv ...")
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.eval()

all_true, all_pred, all_coords = [], [], []
for x, y, coords in tqdm(test_loader, desc='Test set'):
    x = x.unsqueeze(1).to(device)
    logits = model(x)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    all_pred.extend(preds)
    all_true.extend(y.numpy())
    all_coords.append(coords.numpy())

all_coords = np.concatenate(all_coords, axis=0)

cm = confusion_matrix(all_true, all_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'), index=False)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

report = classification_report(all_true, all_pred, target_names=target_names)
with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)
print(report)

save_result = pd.DataFrame({
    'true_label': all_true,
    'pred_label': all_pred,
    'z': all_coords[:, 0],
    'x': all_coords[:, 1],
    'y': all_coords[:, 2]
})
save_result.to_csv(os.path.join(save_dir, 'prediction_result.csv'), index=False)
print("Saved prediction_result.csv, confusion_matrix, and classification_report.")

