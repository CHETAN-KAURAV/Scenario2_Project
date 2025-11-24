"""
train_models_sleep_stage.py
Train models for sleep stage classification (10 epochs for speed).

Usage:
    python train_models_sleep_stage.py --dataset Dataset_SleepStage/sleep_stage_dataset.parquet --model cnn
    python train_models_sleep_stage.py --dataset Dataset_SleepStage/sleep_stage_dataset.parquet --model convlstm
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)


class CNN1D(nn.Module):
    def __init__(self, input_channels=3, num_classes=5, seq_length=960):
        super(CNN1D, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flatten_size = 256 * (seq_length // 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.fc(x)


class ConvLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=5, seq_length=960):
        super(ConvLSTM, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class RespiratoryDataset(Dataset):
    def __init__(self, dataframe, label_encoder=None):
        self.data = dataframe.reset_index(drop=True)
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.data['label'])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        airflow = np.array(row['airflow'])
        thoracic = np.array(row['thoracic'])
        spo2 = np.array(row['spo2'])
        spo2_upsampled = np.interp(np.linspace(0, 1, 960), np.linspace(0, 1, len(spo2)), spo2)
        signals = np.stack([airflow, thoracic, spo2_upsampled])
        return torch.FloatTensor(signals), torch.LongTensor([self.labels[idx]])[0]


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(signals), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = model(signals).max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in dataloader:
            _, predicted = model(signals.to(device)).max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def calculate_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    metrics = {}
    for i, class_name in enumerate(class_names):
        TP, FP, FN = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        TN = cm.sum() - (TP + FP + FN)
        metrics[class_name] = {
            'accuracy': (TP + TN) / cm.sum() if cm.sum() > 0 else 0,
            'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
            'recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0
        }
    return metrics, cm, accuracy_score(y_true, y_pred)


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def leave_one_participant_out_cv(model_class, model_name, dataset_path, results_dir, epochs=10):
    print(f"\n{'=' * 70}\nBONUS: {model_name} Sleep Stage Classification\n{'=' * 70}\n")
    df = pd.read_parquet(dataset_path)
    participants = sorted(df['participant_id'].unique())
    print(f"Participants: {participants}")
    print(f"Total windows: {len(df)}")
    print(f"Sleep Stages:\n{df['label'].value_counts()}\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    os.makedirs(results_dir, exist_ok=True)

    fold_metrics, all_true, all_pred = [], [], []

    for fold_idx, test_participant in enumerate(participants, 1):
        print(f"\n{'─' * 70}\nFOLD {fold_idx}/{len(participants)} - Test: {test_participant}\n{'─' * 70}")
        train_df = df[df['participant_id'] != test_participant]
        test_df = df[df['participant_id'] == test_participant]
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")

        train_dataset = RespiratoryDataset(train_df)
        test_dataset = RespiratoryDataset(test_df, label_encoder=train_dataset.label_encoder)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        num_classes = len(train_dataset.label_encoder.classes_)
        model = model_class(input_channels=3, num_classes=num_classes, seq_length=960).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"Training ({epochs} epochs)...")
        for epoch in range(epochs):
            loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.2f}%")

        print("Evaluating...")
        y_true, y_pred = evaluate(model, test_loader, device)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        class_names = train_dataset.label_encoder.classes_
        metrics, cm, acc = calculate_metrics(y_true, y_pred, class_names)
        fold_metrics.append(metrics)

        cm_path = os.path.join(results_dir, f'{model_name}_SleepStage_fold{fold_idx}_{test_participant}_cm.png')
        plot_confusion_matrix(cm, class_names, f'{model_name} Sleep Stage - Fold {fold_idx} ({test_participant})',
                              cm_path)
        print(f"Fold {fold_idx} Accuracy: {acc * 100:.2f}%")

    print(f"\n{'=' * 70}\nOVERALL SLEEP STAGE RESULTS - {model_name}\n{'=' * 70}\n")
    class_names = train_dataset.label_encoder.classes_
    overall_metrics, overall_cm, overall_acc = calculate_metrics(all_true, all_pred, class_names)

    cm_path = os.path.join(results_dir, f'{model_name}_SleepStage_overall_cm.png')
    plot_confusion_matrix(overall_cm, class_names, f'{model_name} - Sleep Stage Overall', cm_path)

    print(f"Overall Accuracy: {overall_acc * 100:.2f}%\n")
    for class_name in class_names:
        values = [fold[class_name]['recall'] for fold in fold_metrics]
        print(f"{class_name}: {np.mean(values) * 100:.2f}% recall")

    with open(os.path.join(results_dir, f'{model_name}_SleepStage_results.txt'), 'w') as f:
        f.write(f"{model_name} - Sleep Stage Classification Results\n{'=' * 70}\n\n")
        f.write(f"Overall Accuracy: {overall_acc * 100:.2f}%\n\n")
        for class_name in class_names:
            f.write(f"{class_name}:\n")
            for metric in ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity']:
                vals = [fold[class_name][metric] for fold in fold_metrics]
                f.write(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
            f.write("\n")

    print(f"\n✓ Results saved to {results_dir}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True, choices=['cnn', 'convlstm'])
    parser.add_argument('--results_dir', default='Results_SleepStage')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    model_class = CNN1D if args.model == 'cnn' else ConvLSTM
    model_name = 'CNN1D' if args.model == 'cnn' else 'ConvLSTM'

    results_dir = os.path.join(args.results_dir, model_name)
    leave_one_participant_out_cv(model_class, model_name, args.dataset, results_dir, epochs=args.epochs)

    print(f"\n{'=' * 70}\nBONUS TASK COMPLETE!\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
