import sys
import copy
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

# ── Config ────────────────────────────────────────────────────────────────────
START       = int(sys.argv[1])
END         = int(sys.argv[2])
RANDOM_SEED = int(sys.argv[3])

NUM_ITEMS = 10000
BATCH_SIZE = 32
NUM_EPOCHS = 100
DROPOUT_RATE = 0.3
L1_LAMBDA = 0.005
L2_LAMBDA = 0.001
EARLY_STOPPING_PATIENCE = 10


DATASET_PATH = Path('/group/pmc021/amunif/epi-thesis/workflow/17_Pairwise Ranking HepG2 GSE76344/dataset/')
OUTPUT_PATH  = Path('/group/pmc021/amunif/epi-thesis/workflow/17_Pairwise Ranking HepG2 GSE76344/output/')

print(f"Seed: {RANDOM_SEED}  |  Range: {START}-{END}")
print(f"Dataset : {DATASET_PATH}")
print(f"Output  : {OUTPUT_PATH}")

# ── Device & seed ─────────────────────────────────────────────────────────────

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    return device

device = get_device()

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Data loading ──────────────────────────────────────────────────────────────

data_df = pl.read_parquet(DATASET_PATH / 'HepG2_exp_histones.parquet')
permutation_lst = pl.read_parquet(DATASET_PATH / 'marker_combinations.parquet')["combination"].to_list()

# ── Index split ───────────────────────────────────────────────────────────────


all_idx = np.arange(len(data_df))

# First split: 80% train, 20% temp
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
)

# Second split: 50% val, 50% test from temp
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
)

print(f"Train : {len(train_idx):>6}  ({len(train_idx)/len(data_df)*100:.1f}%)")
print(f"Val   : {len(val_idx):>6}  ({len(val_idx)/len(data_df)*100:.1f}%)")
print(f"Test  : {len(test_idx):>6}  ({len(test_idx)/len(data_df)*100:.1f}%)")

# Sanity check – no overlap between partitions
assert len(set(train_idx) & set(val_idx))  == 0, "Train/Val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap!"
assert len(set(val_idx)   & set(test_idx)) == 0, "Val/Test overlap!"
print("No overlap between partitions")

# ── Helper functions ──────────────────────────────────────────────────────────


def generate_pairs_with_labels(data_df, indexes, num_pairs, seed):
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(indexes, size=(num_pairs, 2), replace=True)
    val_1  = data_df["value_1"][idx[:, 0]].to_numpy()
    val_2  = data_df["value_1"][idx[:, 1]].to_numpy()
    labels = (val_1 > val_2).astype(int)
    return np.column_stack([idx[:, 0], idx[:, 1], labels])

def get_label_distribution(pairs, split_name):
    unique, counts = np.unique(pairs[:, 2], return_counts=True)
    total = len(pairs)
    print(f"{split_name} label distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {int(label)}: {count:>6} ({count/total*100:.1f}%)")

def format_file_number(number):
    return f"{number:03d}"

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, item_name, file_name):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_losses,     'blue',   label='Training Loss')
    ax1.plot(epochs, val_losses,        'orange', label='Validation Loss')
    ax1.set_title(f'Training and Validation Loss\n{item_name}', fontsize=10)
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.set_ylim(0.4, 0.8); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_accuracies, 'blue',   label='Training Accuracy')
    ax2.plot(epochs, val_accuracies,   'orange', label='Validation Accuracy')
    ax2.set_title(f'Training and Validation Accuracy\n{item_name}', fontsize=10)
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(50, 90); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# ── Dataset ───────────────────────────────────────────────────────────────────


class LiverDataset(Dataset):
    def __init__(self, total_samples, features, pairs):
        self.total_samples = total_samples
        self.features = features   # numpy array: [gene_id, histone]
        self.pairs    = pairs      # numpy array: [gene_id1, gene_id2, label]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pair                    = self.pairs[idx % len(self.pairs)]
        gene_id1, gene_id2, label = int(pair[0]), int(pair[1]), int(pair[2])
        feature = np.concatenate(
            (self.features[gene_id1, 1], self.features[gene_id2, 1]), axis=0
        )
        return feature, label

# ── Model ─────────────────────────────────────────────────────────────────────
# DirectRanker-style antisymmetric twin-subnet model.
# Each gene's features pass through the same shared subnet (weight-tied),
# producing scalar scores s1 and s2. Output = sigmoid(s1 - s2), so:
#   f(gene1, gene2) + f(gene2, gene1) = 1  (antisymmetry guaranteed)
class BinaryClassifierDropOutL1L2(nn.Module):
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1,
                 dropout_rate=0.2, l1_lambda=0.001, l2_lambda=0.001):
        super().__init__()
        self.gene_input_size = input_size // 2

        self.subnet = nn.Sequential(
            nn.Linear(self.gene_input_size, hidden1_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden1_size, hidden2_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden2_size, output_size),
            # No sigmoid here — applied after subtraction
        )
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        gene1 = x[:, :self.gene_input_size]
        gene2 = x[:, self.gene_input_size:]

        score1  = self.subnet(gene1)
        score2  = self.subnet(gene2)
        outputs = torch.sigmoid(score1 - score2)

        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.linalg.norm(param, 1)
                l2_reg = l2_reg + torch.linalg.norm(param, 2)

        return outputs, self.l1_lambda * l1_reg, self.l2_lambda * l2_reg

    def predict(self, x):
        with torch.no_grad():
            outputs, _, _ = self.forward(x)
            return (outputs >= 0.5).long()

# ── Main experiment loop ──────────────────────────────────────────────────────

for number in range(START, END):
    item        = permutation_lst[number]
    ITEM_NAME   = '-'.join(item)
    FILE_NUMBER = format_file_number(number)
    print(f"\n{FILE_NUMBER}-{ITEM_NAME}")

    # Build feature matrix for this histone combination
    df   = data_df.with_columns(histone=pl.concat_list(item))
    X_np = df["gene_id", "histone"].to_numpy()
    INPUT_SIZE = len(X_np[0][1]) * 2  # concatenated features of two genes

    # Generate pairs and report label balance
    train_pairs = generate_pairs_with_labels(data_df, train_idx, int(0.8 * NUM_ITEMS), RANDOM_SEED)
    val_pairs   = generate_pairs_with_labels(data_df, val_idx,   int(0.1 * NUM_ITEMS), RANDOM_SEED)
    test_pairs  = generate_pairs_with_labels(data_df, test_idx,  int(0.1 * NUM_ITEMS), RANDOM_SEED)
    get_label_distribution(train_pairs, "Train")
    get_label_distribution(val_pairs,   "Val")
    get_label_distribution(test_pairs,  "Test")

    # DataLoaders
    train_loader = DataLoader(LiverDataset(int(0.8 * NUM_ITEMS), X_np, train_pairs),
                              batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(LiverDataset(int(0.1 * NUM_ITEMS), X_np, val_pairs),
                              batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(LiverDataset(int(0.1 * NUM_ITEMS), X_np, test_pairs),
                              batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model, loss, optimiser, scheduler
    model     = BinaryClassifierDropOutL1L2(
                    input_size=INPUT_SIZE, dropout_rate=DROPOUT_RATE,
                    l1_lambda=L1_LAMBDA, l2_lambda=L2_LAMBDA).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # Metric history
    train_losses, val_losses         = [], []
    train_accuracies, val_accuracies = [], []
    train_aucs, val_aucs             = [], []
    val_precisions, val_recalls, val_f1_scores = [], [], []

    # Early stopping state
    best_val_loss      = float('inf')
    early_stop_counter = 0
    best_model_weights = None

    # Output file paths
    prefix           = f"{FILE_NUMBER}-{ITEM_NAME}-seed{RANDOM_SEED}"
    IMAGE_FILE       = OUTPUT_PATH / 'img' / 'train' / f"{prefix}-ranking.png"
    TEST_FILE        = OUTPUT_PATH / 'test'           / f"{prefix}-test-results.txt"
    TRAINING_FILE    = OUTPUT_PATH / 'train'          / f"{prefix}-train-validation-metrics.csv"
    TEST_RESULT_FILE = OUTPUT_PATH / 'test'           / f"{prefix}-test-metrics.csv"
    ROC_FILE         = OUTPUT_PATH / 'img' / 'roc'   / f"{prefix}-roc.png"
    CM_FILE          = OUTPUT_PATH / 'img' / 'cm'    / f"{prefix}-cm.png"

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_true_labels, train_predicted_labels = [], []

        for X_batch, y_batch in train_loader:
            outputs, l1_reg, l2_reg = model(X_batch.to(dtype=torch.float32).to(device))
            loss = criterion(outputs.squeeze(1), y_batch.to(dtype=torch.float32).to(device)) \
                   + l1_reg + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions    = (outputs.squeeze(1) >= 0.5).long()
            train_loss    += loss.item()
            train_correct += (predictions == y_batch.to(device)).sum().item()
            train_total   += y_batch.size(0)
            train_true_labels.extend(y_batch.cpu().numpy())
            train_predicted_labels.extend(predictions.cpu().numpy())

        avg_train_loss  = train_loss / len(train_loader)
        train_accuracy  = 100 * train_correct / train_total
        train_auc_score = roc_auc_score(train_true_labels, train_predicted_labels)

        train_losses.append(round(avg_train_loss, 4))
        train_accuracies.append(round(train_accuracy, 2))
        train_aucs.append(round(train_auc_score, 2))

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_true_labels, val_predicted_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs, l1_reg, l2_reg = model(X_batch.to(dtype=torch.float32).to(device))
                loss = criterion(outputs.squeeze(1), y_batch.to(dtype=torch.float32).to(device)) \
                       + l1_reg + l2_reg

                predictions  = (outputs.squeeze(1) >= 0.5).long()
                val_loss    += loss.item()
                val_correct += (predictions == y_batch.to(dtype=torch.float32).to(device)).sum().item()
                val_total   += y_batch.size(0)
                val_true_labels.extend(y_batch.cpu().numpy())
                val_predicted_labels.extend(predictions.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            val_true_labels, val_predicted_labels, average='binary'
        )
        val_auc_score = roc_auc_score(
            np.array(val_true_labels), np.array(val_predicted_labels)
        )

        val_losses.append(round(avg_val_loss, 4))
        val_accuracies.append(round(val_accuracy, 2))
        val_aucs.append(round(val_auc_score, 2))
        val_precisions.append(round(precision, 2))
        val_recalls.append(round(recall, 2))
        val_f1_scores.append(round(f1_score, 2))

        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
              f'[TRAIN] Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, AUC: {train_auc_score:.2f} '
              f'[VAL] Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, '
              f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}, '
              f'AUC: {val_auc_score:.2f}')

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss      = avg_val_loss
            early_stop_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f'  Early stopping at epoch {epoch+1} '
                      f'(no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)')
                break

    # Restore best weights before evaluation
    model.load_state_dict(best_model_weights)

    # Plot learning curves
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                 ITEM_NAME, IMAGE_FILE)

    # ── Test phase ────────────────────────────────────────────────────────────
    model.eval()
    test_predictions, test_true_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs, _, _ = model(X_batch.to(dtype=torch.float32).to(device))
            predictions   = (outputs.squeeze(1) >= 0.5).long()
            test_true_labels.extend(y_batch.cpu().numpy())
            test_predictions.extend(predictions.cpu().numpy())

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_true_labels, test_predictions, average='binary'
    )
    test_acc_score = accuracy_score(test_true_labels, test_predictions) * 100
    test_auc_score = roc_auc_score(test_true_labels, test_predictions)
    cr = classification_report(test_true_labels, test_predictions)
    cm = confusion_matrix(test_true_labels, test_predictions)

    test_output = (
        f"Seed: {RANDOM_SEED}\n"
        f"Accuracy:  {test_acc_score:.2f} %\n"
        f"AUC:       {test_auc_score:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-Score:  {f1_score:.4f}\n\n"
        f"Detailed Classification Report:\n{cr}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    print(test_output)

    with open(TEST_FILE, 'w', newline='') as f:
        f.write(test_output)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    experiment_pl = pl.DataFrame({
        'seed':          RANDOM_SEED,
        'train_loss':    train_losses,
        'train_accuracy':train_accuracies,
        'train_auc':     train_aucs,
        'val_loss':      val_losses,
        'val_accuracy':  val_accuracies,
        'val_auc':       val_aucs,
    })
    experiment_pl.write_csv(TRAINING_FILE)
    experiment_mean = experiment_pl.mean()

    pl.DataFrame({
        'seed':               RANDOM_SEED,
        'histone_marker':     ITEM_NAME,
        'epochs_trained':     len(train_losses),
        'train_loss_avg':     round(experiment_mean["train_loss"][0],     4),
        'train_accuracy_avg': round(experiment_mean["train_accuracy"][0], 4),
        'train_auc_avg':      round(experiment_mean["train_auc"][0],      4),
        'val_loss_avg':       round(experiment_mean["val_loss"][0],       4),
        'val_accuracy_avg':   round(experiment_mean["val_accuracy"][0],   4),
        'val_auc_avg':        round(experiment_mean["val_auc"][0],        4),
        'test_accuracy':      round(test_acc_score, 4),
        'test_auc':           round(test_auc_score, 4),
    }).write_csv(TEST_RESULT_FILE)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(test_true_labels, test_predictions)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {ITEM_NAME} (seed {RANDOM_SEED})')
    plt.legend()
    plt.savefig(ROC_FILE)
    plt.close()

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display.plot(ax=ax)
    plt.savefig(CM_FILE)
    plt.close()

print("\nBatch FINISHED!!!")
