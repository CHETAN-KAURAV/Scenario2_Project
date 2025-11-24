"""
train_models.py
Train and evaluate 1D CNN and Conv-LSTM models for respiratory event classification.

Usage:
    python train_models.py --dataset Dataset/respiratory_dataset.parquet --model cnn
    python train_models.py --dataset Dataset/respiratory_dataset.parquet --model convlstm
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class CNN1D(nn.Module):
    """1D Convolutional Neural Network"""

    def __init__(self, input_channels=3, num_classes=3, seq_length=960):
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

        # Calculate flattened size
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
        x = self.fc(x)
        return x


class ConvLSTM(nn.Module):
    """1D Conv-LSTM Network"""

    def __init__(self, input_channels=3, num_classes=3, seq_length=960):
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

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Classifier (512 because bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Prepare for LSTM: (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last time step
        x = lstm_out[:, -1, :]

        # Classifier
        x = self.fc(x)
        return x


# ============================================================================
# DATASET CLASS
# ============================================================================

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

        # Stack channels: airflow (960), thoracic (960), spo2 (120)
        # Need to upsample spo2 to 960
        airflow = np.array(row['airflow'])
        thoracic = np.array(row['thoracic'])
        spo2 = np.array(row['spo2'])

        # Upsample spo2 from 120 to 960
        spo2_upsampled = np.interp(
            np.linspace(0, 1, 960),
            np.linspace(0, 1, len(spo2)),
            spo2
        )

        # Stack as (3, 960)
        signals = np.stack([airflow, thoracic, spo2_upsampled])

        return torch.FloatTensor(signals), torch.LongTensor([self.labels[idx]])[0]


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate per-class metrics"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    metrics = {}
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        accuracy = (TP + TN) / cm.sum() if cm.sum() > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity = recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    overall_accuracy = accuracy_score(y_true, y_pred)
    return metrics, cm, overall_accuracy


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================================
# LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION
# ============================================================================

def leave_one_participant_out_cv(model_class, model_name, dataset_path, results_dir):
    print(f"\n{'=' * 70}")
    print(f"Training {model_name} with Leave-One-Participant-Out CV")
    print(f"{'=' * 70}\n")

    # Load dataset
    df = pd.read_parquet(dataset_path)
    participants = sorted(df['participant_id'].unique())

    print(f"Participants: {participants}")
    print(f"Total windows: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(results_dir, exist_ok=True)

    fold_metrics = []
    all_true = []
    all_pred = []

    for fold_idx, test_participant in enumerate(participants):
        print(f"\n{'─' * 70}")
        print(f"FOLD {fold_idx + 1}/{len(participants)} - Test: {test_participant}")
        print(f"{'─' * 70}")

        # Split data
        train_df = df[df['participant_id'] != test_participant]
        test_df = df[df['participant_id'] == test_participant]

        print(f"Train windows: {len(train_df)} | Test windows: {len(test_df)}")

        # Create datasets
        train_dataset = RespiratoryDataset(train_df)
        test_dataset = RespiratoryDataset(test_df, label_encoder=train_dataset.label_encoder)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        num_classes = len(train_dataset.label_encoder.classes_)
        model = model_class(input_channels=3, num_classes=num_classes, seq_length=960).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training
        print("Training...")
        for epoch in range(50):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/50 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Evaluation
        print("Evaluating...")
        y_true, y_pred = evaluate(model, test_loader, device)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        # Calculate metrics
        class_names = train_dataset.label_encoder.classes_
        metrics, cm, acc = calculate_metrics(y_true, y_pred, class_names)
        fold_metrics.append(metrics)

        # Save confusion matrix
        cm_path = os.path.join(results_dir, f'{model_name}_fold{fold_idx + 1}_{test_participant}_cm.png')
        plot_confusion_matrix(cm, class_names,
                              f'{model_name} - Fold {fold_idx + 1} ({test_participant})', cm_path)

        print(f"Fold {fold_idx + 1} Accuracy: {acc * 100:.2f}%")

    # Overall results
    print(f"\n{'=' * 70}")
    print(f"OVERALL RESULTS - {model_name}")
    print(f"{'=' * 70}\n")

    class_names = train_dataset.label_encoder.classes_
    overall_metrics, overall_cm, overall_acc = calculate_metrics(all_true, all_pred, class_names)

    # Save overall confusion matrix
    cm_path = os.path.join(results_dir, f'{model_name}_overall_cm.png')
    plot_confusion_matrix(overall_cm, class_names, f'{model_name} - Overall', cm_path)

    # Calculate mean and std across folds
    print("Per-Class Metrics (Mean ± Std across folds):\n")
    for class_name in class_names:
        print(f"{class_name}:")
        for metric_name in ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity']:
            values = [fold[class_name][metric_name] for fold in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")
        print()

    # Save results to file
    with open(os.path.join(results_dir, f'{model_name}_results.txt'), 'w') as f:
        f.write(f"{model_name} - Leave-One-Participant-Out CV Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Per-Class Metrics (Mean ± Std across folds):\n\n")
        for class_name in class_names:
            f.write(f"{class_name}:\n")
            for metric_name in ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity']:
                values = [fold[class_name][metric_name] for fold in fold_metrics]
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"  {metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write("\n")

        f.write(f"\nOverall Accuracy: {overall_acc * 100:.2f}%\n")

    print(f"✓ Results saved to {results_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset parquet file')
    parser.add_argument('--model', required=True, choices=['cnn', 'convlstm'],
                        help='Model architecture')
    parser.add_argument('--results_dir', default='Results', help='Directory to save results')
    args = parser.parse_args()

    if args.model == 'cnn':
        model_class = CNN1D
        model_name = 'CNN1D'
    else:
        model_class = ConvLSTM
        model_name = 'ConvLSTM'

    results_dir = os.path.join(args.results_dir, model_name)
    leave_one_participant_out_cv(model_class, model_name, args.dataset, results_dir)

    print("\n" + "=" * 70)
    print("Why Leave-One-Participant-Out CV?")
    print("=" * 70)
    print("""
Random 80-20 split would cause DATA LEAKAGE:
- Windows from the same participant in both train and test sets
- Model would learn participant-specific patterns, not generalizable features
- Test performance would be artificially inflated
- Poor performance on new, unseen participants

Leave-One-Participant-Out CV:
- Ensures no participant appears in both train and test
- Tests true generalization to new individuals
- More realistic evaluation for clinical deployment
- Standard practice for physiological/personalized data
    """)


if __name__ == "__main__":
    main()
