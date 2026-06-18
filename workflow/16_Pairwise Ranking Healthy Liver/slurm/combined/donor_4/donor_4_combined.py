# %%
# Import libraries
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

# %%
# Config
START       = int(sys.argv[1])
END         = int(sys.argv[2])
RANDOM_SEED = int(sys.argv[3])

# Or this one for testing
# START       = 0
# END         = 1
# RANDOM_SEED = 42

NUM_ITEMS = 10_000
BATCH_SIZE = 32
NUM_EPOCHS = 100
DROPOUT_RATE = 0.3
L1_LAMBDA = 0.005
L2_LAMBDA = 0.001
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4
ANTISYM_LAMBDA = 0.5

# %%
# Folder setup
DATASET_PATH = Path('/group/pmc021/amunif/epi-thesis/workflow/16_Pairwise Ranking Healthy Liver/dataset/donor_4')
OUTPUT_PATH  = Path('/group/pmc021/amunif/epi-thesis/workflow/16_Pairwise Ranking Healthy Liver/output/combined/donor_4')

# Ensure output subfolders exist
for sub in ['img/train', 'img/roc', 'img/cm', 'test', 'train']:
    (OUTPUT_PATH / sub).mkdir(parents=True, exist_ok=True)

print(f"Seed: {RANDOM_SEED}  |  Range: {START}-{END}")
print(f"Dataset : {DATASET_PATH}")
print(f"Output  : {OUTPUT_PATH}")

# %%
# Setup device and seed
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    return device

device = get_device()
PIN_MEMORY = device.type == "cuda"

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %%
# Data Loading
data_df = pl.read_parquet(DATASET_PATH / 'donor4_exp_histones.parquet')
permutation_lst = pl.read_parquet(DATASET_PATH / 'marker_combinations.parquet')["combination"].to_list()

# %%
# Index split
all_idx = np.arange(len(data_df))

train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
)

print(f"Train : {len(train_idx):>6}  ({len(train_idx)/len(data_df)*100:.1f}%)")
print(f"Val   : {len(val_idx):>6}  ({len(val_idx)/len(data_df)*100:.1f}%)")
print(f"Test  : {len(test_idx):>6}  ({len(test_idx)/len(data_df)*100:.1f}%)")

assert len(set(train_idx) & set(val_idx))  == 0, "Train/Val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap!"
assert len(set(val_idx)   & set(test_idx)) == 0, "Val/Test overlap!"
print("No overlap between partitions")

# %%
# Helper function

def format_file_number(number):
    return f"{number:03d}"

# Generate pairs for DirectRanker
def generate_pairs_with_labels(data_df, indexes, num_pairs, seed):
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(indexes, size=(num_pairs, 2), replace=True)
    val_1  = data_df["value"][idx[:, 0]].to_numpy()
    val_2  = data_df["value"][idx[:, 1]].to_numpy()
    labels = (val_1 > val_2).astype(int)
    return np.column_stack([idx[:, 0], idx[:, 1], labels])

# Generate pairs for sklearn based algorithm
def pairs_to_baseline_arrays(X_np, pairs):
    idx1 = pairs[:, 0].astype(int)
    idx2 = pairs[:, 1].astype(int)
    y    = pairs[:, 2].astype(int)

    feats1 = np.stack(X_np[idx1, 1])
    feats2 = np.stack(X_np[idx2, 1])
    X = np.concatenate([feats1, feats2], axis=1)
    return X, y

# Get the label distributions
def get_label_distribution(pairs, split_name):
    unique, counts = np.unique(pairs[:, 2], return_counts=True)
    total = len(pairs)
    print(f"{split_name} label distribution:")
    dist = {0: 0, 1: 0}
    for label, count in zip(unique, counts):
        print(f"  Label {int(label)}: {count:>6} ({count/total*100:.1f}%)")
        dist[int(label)] = int(count)
    return dist

# Plot the metrics
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

# Calculate the antisymmetry score for DirectRanker
def antisymmetry_score_torch(model, X_test_tensor, device, confidence_threshold=0.05):
    model.eval()
    half = X_test_tensor.shape[1] // 2
    X_swapped = torch.cat([X_test_tensor[:, half:], X_test_tensor[:, :half]], dim=1) # Swap the tensor

    with torch.no_grad():
        out_orig,    _, _ = model(X_test_tensor.to(dtype=torch.float32).to(device))
        out_swapped, _, _ = model(X_swapped.to(dtype=torch.float32).to(device))

    prob_orig    = out_orig.squeeze(1).cpu()
    prob_swapped = out_swapped.squeeze(1).cpu()
    pred_orig    = (prob_orig    >= 0.5).long()
    pred_swapped = (prob_swapped >= 0.5).long()

    antisym_mask     = (pred_orig + pred_swapped == 1)
    antisym_fraction = antisym_mask.float().mean().item()

    confident_mask = (prob_orig - 0.5).abs() >= confidence_threshold
    confident_antisym = (antisym_mask[confident_mask].float().mean().item()
                         if confident_mask.sum() > 0 else float('nan'))

    near_boundary   = (~confident_mask).sum().item()
    violations      = (~antisym_mask).sum().item()
    violation_probs = (prob_orig - 0.5).abs()[~antisym_mask]

    print(f"  Antisymmetry score (raw):        {antisym_fraction:.4f}")
    print(f"  Antisymmetry score (confident):  {confident_antisym:.4f}")
    print(f"  Total violations:                {violations} / {len(pred_orig)}")
    print(f"  Pairs near boundary:             {near_boundary} ({near_boundary/len(pred_orig)*100:.1f}%)")
    if len(violation_probs) > 0:
        print(f"  Avg |prob-0.5| at violation:     {violation_probs.mean():.4f}")

    return antisym_fraction

# Antisymmetric score for sklearn based
def antisymmetry_score_sklearn(clf, X_test):
    half = X_test.shape[1] // 2
    X_swapped = np.concatenate([X_test[:, half:], X_test[:, :half]], axis=1)
    pred_orig    = clf.predict(X_test)
    pred_swapped = clf.predict(X_swapped)
    return float(np.mean(pred_orig + pred_swapped == 1))

# %%
# Dataset Class for DirectRanker
class LiverDataset(Dataset):
    def __init__(self, total_samples, features, pairs):
        self.total_samples = total_samples
        self.features = features
        self.pairs    = pairs

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        gene_id1, gene_id2, label = int(pair[0]), int(pair[1]), int(pair[2])
        feature = np.concatenate(
            (self.features[gene_id1, 1], self.features[gene_id2, 1]), axis=0
        )
        return feature, label

# %%
# DirectRanker Model
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
        )
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        gene1 = x[:, :self.gene_input_size]
        gene2 = x[:, self.gene_input_size:]

        score1  = self.subnet(gene1)
        score2  = self.subnet(gene2)
        outputs = torch.sigmoid(score1 - score2) # Calculate the difference

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

# %%
# Baseline Classifier
def build_baseline_classifiers(random_state):
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                random_state=random_state, n_jobs=-1,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None, min_samples_leaf=4,
                random_state=random_state, n_jobs=-1,
            )),
        ]),
        "SVM_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=random_state, dual="auto"),
                cv=3, method="sigmoid",
            )),
        ]),
    }

# %%
# Evaluate the baseline
def evaluate_baseline(clf, X_test, y_test, clf_name, item_name):
    y_pred  = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") \
              else clf.decision_function(X_test)

    acc  = accuracy_score(y_test, y_pred) * 100
    auc_ = roc_auc_score(y_test, y_score)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    antisym = antisymmetry_score_sklearn(clf, X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  [{clf_name}]  Accuracy: {acc:.2f}%  AUC: {auc_:.4f}  "
          f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  "
          f"Antisym: {antisym:.4f}")

    return {
        "model":          clf_name,
        "seed":           RANDOM_SEED,
        "histone_marker": item_name,
        "test_accuracy":  round(acc, 4),
        "test_auc":       round(auc_, 4),
        "test_precision":      round(prec, 4),
        "test_recall":         round(rec, 4),
        "test_f1":             round(f1, 4),
        "antisymmetry":   round(antisym, 4),
    }, cm, y_pred, y_score

# %%
# Main experiment loop
for number in range(START, END):
    item        = permutation_lst[number]
    ITEM_NAME   = '-'.join(item)
    FILE_NUMBER = format_file_number(number)
    print(f"\n{'='*70}\n{FILE_NUMBER}-{ITEM_NAME}\n{'='*70}")

    # Build feature matrix (shared by DirectRanker and baselines)
    df   = data_df.with_columns(histone=pl.concat_list(item))
    X_np = df["gene_id", "histone"].to_numpy()
    INPUT_SIZE = len(X_np[0][1]) * 2

    # Generate IDENTICAL pairs for DirectRanker and baselines
    train_pairs = generate_pairs_with_labels(data_df, train_idx, int(0.8 * NUM_ITEMS), RANDOM_SEED)
    val_pairs   = generate_pairs_with_labels(data_df, val_idx,   int(0.1 * NUM_ITEMS), RANDOM_SEED)
    test_pairs  = generate_pairs_with_labels(data_df, test_idx,  int(0.1 * NUM_ITEMS), RANDOM_SEED)
    train_dist = get_label_distribution(train_pairs, "Train")
    val_dist   = get_label_distribution(val_pairs,   "Val")
    test_dist  = get_label_distribution(test_pairs,  "Test")

    # Baseline arrays derived from the same pairs (fair comparison)
    X_train_bl, y_train_bl = pairs_to_baseline_arrays(X_np, train_pairs)
    X_val_bl,   y_val_bl   = pairs_to_baseline_arrays(X_np, val_pairs)
    X_test_bl,  y_test_bl  = pairs_to_baseline_arrays(X_np, test_pairs)

    # Output file paths (shared prefix across DirectRanker and baselines)
    prefix           = f"{FILE_NUMBER}-{ITEM_NAME}-seed{RANDOM_SEED}"
    IMAGE_FILE       = OUTPUT_PATH / 'img' / 'train' / f"{prefix}-ranking.png"
    TEST_FILE        = OUTPUT_PATH / 'test'          / f"{prefix}-test-results.txt"
    TRAINING_FILE    = OUTPUT_PATH / 'train'         / f"{prefix}-train-validation-metrics.csv"
    TEST_RESULT_FILE = OUTPUT_PATH / 'test'          / f"{prefix}-test-metrics.csv"
    ROC_FILE         = OUTPUT_PATH / 'img' / 'roc'   / f"{prefix}-roc-combined.png"
    CM_DIR           = OUTPUT_PATH / 'img' / 'cm'
    LABEL_DIST_FILE  = OUTPUT_PATH / 'test'          / f"{prefix}-label-distribution.csv"

    # Save label distribution
    label_dist_rows = []
    for split_name, dist, pairs in [
        ("Train", train_dist, train_pairs),
        ("Val",   val_dist,   val_pairs),
        ("Test",  test_dist,  test_pairs),
    ]:
        total = len(pairs)
        for label in (0, 1):
            count = dist[label]
            label_dist_rows.append({
                'seed':           RANDOM_SEED,
                'histone_marker': ITEM_NAME,
                'split':          split_name,
                'label':          label,
                'count':          count,
                'total':          total,
                'percentage':     round(count / total * 100, 2),
            })
    pl.DataFrame(label_dist_rows).write_csv(LABEL_DIST_FILE)
    print(f"Saved label distribution: {LABEL_DIST_FILE}")

    # ════════════════════════════════════════════════════════════════════════
    # PART 1 — DirectRanker training
    # ════════════════════════════════════════════════════════════════════════
    train_loader = DataLoader(LiverDataset(int(0.8 * NUM_ITEMS), X_np, train_pairs),
                              batch_size=BATCH_SIZE, shuffle=True,  pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(LiverDataset(int(0.1 * NUM_ITEMS), X_np, val_pairs),
                              batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(LiverDataset(int(0.1 * NUM_ITEMS), X_np, test_pairs),
                              batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    model     = BinaryClassifierDropOutL1L2(
                    input_size=INPUT_SIZE, dropout_rate=DROPOUT_RATE,
                    l1_lambda=L1_LAMBDA, l2_lambda=L2_LAMBDA).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    train_losses, val_losses         = [], []
    train_accuracies, val_accuracies = [], []
    train_aucs, val_aucs             = [], []
    val_precisions, val_recalls, val_f1_scores = [], [], []

    best_val_loss      = float('inf')
    early_stop_counter = 0
    best_model_weights = None

    print("\n--- Training DirectRanker ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_scores = []
        train_true_labels, train_predicted_labels = [], []

        for X_batch, y_batch in train_loader:
            outputs, l1_reg, l2_reg = model(X_batch.to(dtype=torch.float32).to(device))

            half = X_batch.shape[1] // 2
            X_swapped = torch.cat([X_batch[:, half:], X_batch[:, :half]], dim=1)
            out_swapped, _, _ = model(X_swapped.to(dtype=torch.float32).to(device))

            antisym_loss = ((outputs.squeeze(1) + out_swapped.squeeze(1) - 1.0) ** 2).mean()

            loss = criterion(outputs.squeeze(1), y_batch.to(dtype=torch.float32).to(device)) \
                  + l1_reg + l2_reg + ANTISYM_LAMBDA * antisym_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions    = (outputs.squeeze(1) >= 0.5).long()
            train_loss    += loss.item()
            train_correct += (predictions == y_batch.to(device)).sum().item()
            train_total   += y_batch.size(0)
            train_true_labels.extend(y_batch.cpu().numpy())
            train_predicted_labels.extend(predictions.cpu().numpy())
            train_scores.extend(outputs.squeeze(1).detach().cpu().numpy())

        avg_train_loss  = train_loss / len(train_loader)
        train_accuracy  = 100 * train_correct / train_total
        train_auc_score = roc_auc_score(train_true_labels, train_scores)

        train_losses.append(round(avg_train_loss, 4))
        train_accuracies.append(round(train_accuracy, 2))
        train_aucs.append(round(train_auc_score, 2))

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_true_labels, val_predicted_labels = [], []

        with torch.no_grad():
            val_scores = []
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
                val_scores.extend(outputs.squeeze(1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            val_true_labels, val_predicted_labels, average='binary'
        )
        val_auc_score = roc_auc_score(np.array(val_true_labels), np.array(val_scores))

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

        if avg_val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss      = avg_val_loss
            early_stop_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f'  Early stopping at epoch {epoch+1} '
                      f'(no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)')
                break

    model.load_state_dict(best_model_weights)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                ITEM_NAME, IMAGE_FILE)

    # DirectRanker test phase
    model.eval()
    dr_test_predictions, dr_test_true_labels, dr_test_scores = [], [], []
    test_X_batches = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs, _, _ = model(X_batch.to(dtype=torch.float32).to(device))
            predictions   = (outputs.squeeze(1) >= 0.5).long()
            dr_test_true_labels.extend(y_batch.cpu().numpy())
            dr_test_predictions.extend(predictions.cpu().numpy())
            dr_test_scores.extend(outputs.squeeze(1).cpu().numpy())
            test_X_batches.append(X_batch)

    X_test_all = torch.cat(test_X_batches, dim=0)

    dr_precision, dr_recall, dr_f1, _ = precision_recall_fscore_support(
        dr_test_true_labels, dr_test_predictions, average='binary'
    )
    dr_acc     = accuracy_score(dr_test_true_labels, dr_test_predictions) * 100
    dr_auc     = roc_auc_score(dr_test_true_labels, dr_test_scores)
    dr_cr      = classification_report(dr_test_true_labels, dr_test_predictions)
    dr_cm      = confusion_matrix(dr_test_true_labels, dr_test_predictions)
    dr_antisym = antisymmetry_score_torch(model, X_test_all, device)

    test_output = (
        f"Seed: {RANDOM_SEED}\n"
        f"=== DirectRanker ===\n"
        f"Accuracy:  {dr_acc:.2f} %\n"
        f"AUC:       {dr_auc:.4f}\n"
        f"Precision: {dr_precision:.4f}\n"
        f"Recall:    {dr_recall:.4f}\n"
        f"F1-Score:  {dr_f1:.4f}\n"
        f"Antisymmetry score:  {dr_antisym:.4f}  (1.0 = perfect, ~0.5 = random)\n\n"
        f"Detailed Classification Report:\n{dr_cr}\n\n"
        f"Confusion Matrix:\n{dr_cm}\n"
    )

    # Save per-epoch training history (DirectRanker only)
    experiment_pl = pl.DataFrame({
        'seed':           RANDOM_SEED,
        'train_loss':     train_losses,
        'train_accuracy': train_accuracies,
        'train_auc':      train_aucs,
        'val_loss':       val_losses,
        'val_accuracy':   val_accuracies,
        'val_auc':        val_aucs,
    })
    experiment_pl.write_csv(TRAINING_FILE)

    # ════════════════════════════════════════════════════════════════════════
    # PART 2 — Baseline classifiers (same pairs, same test set)
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- Training baselines (LogisticRegression, RandomForest, SVM) ---")
    baseline_classifiers = build_baseline_classifiers(RANDOM_SEED)

    baseline_results  = []
    baseline_cms       = {}
    baseline_preds      = {}
    baseline_scores     = {}
    fitted_baselines    = {}

    for clf_name, clf in baseline_classifiers.items():
        print(f"\n  Training {clf_name}...")
        clf.fit(X_train_bl, y_train_bl)

        y_val_pred  = clf.predict(X_val_bl)
        y_val_score = (clf.predict_proba(X_val_bl)[:, 1]
                       if hasattr(clf, "predict_proba")
                       else clf.decision_function(X_val_bl))

        val_acc = accuracy_score(y_val_bl, y_val_pred) * 100
        val_auc = roc_auc_score(y_val_bl, y_val_score)
        print(f"  Val accuracy: {val_acc:.2f}%  |  Val AUC: {val_auc:.4f}")

        result, cm, y_pred, y_score = evaluate_baseline(clf, X_test_bl, y_test_bl, clf_name, ITEM_NAME)
        result['val_accuracy'] = round(val_acc, 4)
        result['val_auc'] = round(val_auc, 4)
        baseline_results.append(result)
        baseline_cms[clf_name]    = cm
        baseline_preds[clf_name]  = y_pred
        baseline_scores[clf_name] = y_score
        fitted_baselines[clf_name] = clf

        test_output += (
            f"\n=== {clf_name} ===\n"
            f"Accuracy:  {result['test_accuracy']:.2f} %\n"
            f"AUC:       {result['test_auc']:.4f}\n"
            f"Precision: {result['test_precision']:.4f}\n"
            f"Recall:    {result['test_recall']:.4f}\n"
            f"F1-Score:  {result['test_f1']:.4f}\n"
            f"Antisymmetry score:  {result['antisymmetry']:.4f}\n"
            f"Confusion Matrix:\n{cm}\n"
        )

    print(test_output)
    with open(TEST_FILE, 'w', newline='') as f:
        f.write(test_output)

    # ════════════════════════════════════════════════════════════════════════
    # PART 3 — Combined test-metrics CSV (DirectRanker and all baselines in one file)
    # ════════════════════════════════════════════════════════════════════════
    best_epoch_idx = val_losses.index(min(val_losses))

    combined_rows = [{
        'model':          'DirectRanker',
        'seed':           RANDOM_SEED,
        'histone_marker': ITEM_NAME,
        'epochs_trained': len(train_losses),
        'val_accuracy':   round(val_accuracies[best_epoch_idx], 4),
        'val_auc':        round(val_aucs[best_epoch_idx], 4),
        'test_accuracy':  round(dr_acc, 4),
        'test_auc':       round(dr_auc, 4),
        'test_precision':      round(dr_precision, 4),
        'test_recall':         round(dr_recall, 4),
        'test_f1':             round(dr_f1, 4),
        'antisymmetry':   round(dr_antisym, 4),
    }]
    for r in baseline_results:
        r_full = dict(r)
        r_full['epochs_trained'] = None    # baselines don't train over epochs
        combined_rows.append(r_full)

    pl.DataFrame(combined_rows).write_csv(TEST_RESULT_FILE)
    print(f"Saved combined test metrics: {TEST_RESULT_FILE}")

    # ════════════════════════════════════════════════════════════════════════
    # PART 4 — Combined ROC plot (DirectRanker and 3 baselines in one figure)
    # ════════════════════════════════════════════════════════════════════════
    plt.figure(figsize=(7, 6))

    fpr_dr, tpr_dr, _ = roc_curve(dr_test_true_labels, dr_test_scores)
    roc_auc_dr = auc(fpr_dr, tpr_dr)
    plt.plot(fpr_dr, tpr_dr, lw=2.5, color='black',
             label=f"DirectRanker (AUC = {roc_auc_dr:.3f})")

    for clf_name in fitted_baselines:
        fpr, tpr, _ = roc_curve(y_test_bl, baseline_scores[clf_name])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{clf_name} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — DirectRanker vs Baselines\n{ITEM_NAME} (seed {RANDOM_SEED})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined ROC plot: {ROC_FILE}")

    # ════════════════════════════════════════════════════════════════════════
    # PART 5 — Confusion matrices (one PNG per model, same prefix)
    # ════════════════════════════════════════════════════════════════════════
    all_cms = {'DirectRanker': dr_cm, **baseline_cms}
    for model_name, cm in all_cms.items():
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_display.plot(ax=ax, cmap='Blues')
        ax.set_title(f"{model_name} — {ITEM_NAME}\n(seed {RANDOM_SEED})", fontsize=10)
        plt.tight_layout()
        plt.savefig(CM_DIR / f"{prefix}-cm-{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nFinished {FILE_NUMBER}-{ITEM_NAME} (seed {RANDOM_SEED})")

# %%
print("\nBatch FINISHED!!!")