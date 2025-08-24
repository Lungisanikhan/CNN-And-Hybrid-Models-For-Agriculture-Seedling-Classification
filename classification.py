import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# ==========================
# DEVICE DETECTION
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# DATA PATHS
# ==========================
train_dir = "D:/Deep learning/Deep learning/train/train"
test_dir  = "D:/Deep learning/Deep learning/test/test"

# ==========================
# IMAGE TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==========================
# LOAD TRAINING DATA WITH PROPER TRAIN/VAL SPLIT
# ==========================
def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))

# Load full training dataset
full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform, is_valid_file=is_image_file)

# Split training data into train and validation sets (80/20 split)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ==========================
# TEST DATA (unlabeled - just for prediction)
# ==========================
class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [f for f in os.listdir(folder) if is_image_file(f)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

# ==========================
# CNN MODEL (with memory-efficient feature extraction)
# ==========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes, activation=nn.ReLU()):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            activation,
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to fixed small size
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*4*4, 256)  # Much smaller feature size
        self.dropout = nn.Dropout(0.5)
        self.relu = activation
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return self.fc2(x)
    
    def get_features(self, x):
        """Extract features for traditional ML models"""
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x  # Return features before final classification layer

# ==========================
# TRAIN CNN MODEL
# ==========================
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Training CNN Model ===")
epochs = 5
best_val_acc = 0
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save best model (optional)
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

# ==========================
# MEMORY-EFFICIENT FEATURE EXTRACTION
# ==========================
def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            # Use the dedicated feature extraction method
            feats = model.get_features(images)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)

# Extract features for training traditional ML models
print("\n=== Extracting Features ===")
train_feats, train_labels = extract_features(model, train_loader)
print(f"Training features shape: {train_feats.shape}")

# ==========================
# TRAIN TRADITIONAL ML MODELS (with memory optimization)
# ==========================
print("\n=== Training Traditional ML Models ===")

# SVM with reduced complexity
svm = SVC(kernel='linear', probability=True, random_state=42, C=0.1)  # Lower C for regularization
svm.fit(train_feats, train_labels)

# XGBoost with memory-efficient settings
xgb = XGBClassifier(
    n_estimators=50,  # Reduced from 100
    max_depth=6,      # Limit tree depth
    learning_rate=0.1,
    subsample=0.8,    # Use 80% of samples per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42,
    n_jobs=1,         # Single thread to save memory
    tree_method='approx'  # Memory-efficient tree construction
)

try:
    xgb.fit(train_feats, train_labels)
    xgb_trained = True
except Exception as e:
    print(f"XGBoost training failed due to memory: {e}")
    print("Continuing without XGBoost...")
    xgb_trained = False

# ==========================
# EVALUATION ON VALIDATION SET (during development)
# ==========================
def evaluate_model_on_validation(model, loader, model_name):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    
    print(f"\n=== {model_name} Validation Results ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    return acc, f1, y_true, y_pred, y_prob

# Evaluate CNN on validation set
cnn_acc, cnn_f1, val_labels, cnn_preds, cnn_probs = evaluate_model_on_validation(model, val_loader, "Baseline CNN")

# Extract validation features for traditional ML models
val_feats, val_labels_array = extract_features(model, val_loader)

# Evaluate SVM on validation set
svm_preds = svm.predict(val_feats)
svm_probs = svm.predict_proba(val_feats)

print("\n=== CNN + SVM Validation Results ===")
print(classification_report(val_labels_array, svm_preds, target_names=class_names))
svm_acc = accuracy_score(val_labels_array, svm_preds)
svm_f1 = f1_score(val_labels_array, svm_preds, average="weighted")

# Evaluate XGBoost on validation set (if trained successfully)
if xgb_trained:
    xgb_preds = xgb.predict(val_feats)
    xgb_probs = xgb.predict_proba(val_feats)

    print("\n=== CNN + XGBoost Validation Results ===")
    print(classification_report(val_labels_array, xgb_preds, target_names=class_names))
    xgb_acc = accuracy_score(val_labels_array, xgb_preds)
    xgb_f1 = f1_score(val_labels_array, xgb_preds, average="weighted")
else:
    print("\n=== XGBoost skipped due to memory constraints ===")
    xgb_preds = None
    xgb_probs = None
    xgb_acc = 0
    xgb_f1 = 0

# ==========================
# PREDICTIONS ON TEST SET (unlabeled)
# ==========================
def predict_test_set(model, loader, model_name):
    model.eval()
    predictions = []
    confidences = []
    filenames = []
    
    with torch.no_grad():
        for images, fnames in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
            filenames.extend(fnames)
    
    # Convert predictions to class names
    predicted_classes = [class_names[pred] for pred in predictions]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'filename': filenames,
        'predicted_class': predicted_classes,
        'confidence': confidences
    })
    
    print(f"\n=== {model_name} Test Set Predictions ===")
    print(f"Total test samples: {len(results_df)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print("\nPrediction distribution:")
    print(results_df['predicted_class'].value_counts())
    
    return results_df

# Get CNN predictions on test set
cnn_test_results = predict_test_set(model, test_loader, "Baseline CNN")

# Extract test features for traditional ML models
def extract_test_features(model, loader):
    model.eval()
    features = []
    filenames = []
    with torch.no_grad():
        for images, fnames in loader:
            images = images.to(device)
            # Use the dedicated feature extraction method
            feats = model.get_features(images)
            features.append(feats.cpu().numpy())
            filenames.extend(fnames)
    return np.vstack(features), filenames

test_feats, test_filenames = extract_test_features(model, test_loader)

# SVM predictions on test set
svm_test_preds = svm.predict(test_feats)
svm_test_probs = svm.predict_proba(test_feats)
svm_test_classes = [class_names[pred] for pred in svm_test_preds]
svm_test_conf = np.max(svm_test_probs, axis=1)

svm_test_results = pd.DataFrame({
    'filename': test_filenames,
    'predicted_class': svm_test_classes,
    'confidence': svm_test_conf
})

print("\n=== CNN + SVM Test Set Predictions ===")
print(f"Total test samples: {len(svm_test_results)}")
print(f"Average confidence: {np.mean(svm_test_conf):.4f}")
print("\nPrediction distribution:")
print(svm_test_results['predicted_class'].value_counts())

# XGBoost predictions on test set (if trained)
if xgb_trained:
    xgb_test_preds = xgb.predict(test_feats)
    xgb_test_probs = xgb.predict_proba(test_feats)
    xgb_test_classes = [class_names[pred] for pred in xgb_test_preds]
    xgb_test_conf = np.max(xgb_test_probs, axis=1)

    xgb_test_results = pd.DataFrame({
        'filename': test_filenames,
        'predicted_class': xgb_test_classes,
        'confidence': xgb_test_conf
    })

    print("\n=== CNN + XGBoost Test Set Predictions ===")
    print(f"Total test samples: {len(xgb_test_results)}")
    print(f"Average confidence: {np.mean(xgb_test_conf):.4f}")
    print("\nPrediction distribution:")
    print(xgb_test_results['predicted_class'].value_counts())
else:
    xgb_test_results = pd.DataFrame({
        'filename': test_filenames,
        'predicted_class': ['N/A'] * len(test_filenames),
        'confidence': [0.0] * len(test_filenames)
    })

# ==========================
# ROC CURVES (using validation set)
# ==========================
if num_classes > 2:
    y_true_bin = label_binarize(val_labels_array, classes=list(range(num_classes)))
    
    plt.figure(figsize=(12, 8))
    
    # Micro-average ROC curve
    plt.subplot(1, 2, 1)
    models_to_plot = ["CNN", "SVM"]
    probs_to_plot = [cnn_probs, svm_probs]
    
    if xgb_trained:
        models_to_plot.append("XGB")
        probs_to_plot.append(xgb_probs)
    
    for name, probs in zip(models_to_plot, probs_to_plot):
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(probs).ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (Micro-avg AUC={roc_auc:.2f})")
    
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Micro Average (Validation Set)")
    plt.legend()
    
    # Prediction confidence distribution
    plt.subplot(1, 2, 2)
    plt.hist(cnn_test_results['confidence'], alpha=0.7, label='CNN', bins=20)
    plt.hist(svm_test_results['confidence'], alpha=0.7, label='SVM', bins=20)
    if xgb_trained:
        plt.hist(xgb_test_results['confidence'], alpha=0.7, label='XGBoost', bins=20)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Test Set Prediction Confidence Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================
# FINAL COMPARISON TABLE (Validation Performance)
# ==========================
model_names = ["Baseline CNN", "CNN+SVM"]
val_accuracies = [cnn_acc, svm_acc]
val_f1_scores = [cnn_f1, svm_f1]

if xgb_trained:
    model_names.append("CNN+XGBoost")
    val_accuracies.append(xgb_acc)
    val_f1_scores.append(xgb_f1)

results = pd.DataFrame({
    "Model": model_names,
    "Validation Accuracy": val_accuracies,
    "Validation F1-Score": val_f1_scores
})

print("\n=== Validation Results Comparison ===")
print(results.round(4))

# Print summary
print(f"\n=== Summary ===")
print(f"Best performing model on validation: {results.loc[results['Validation Accuracy'].idxmax(), 'Model']}")
print(f"Best validation accuracy: {results['Validation Accuracy'].max():.4f}")
print(f"Best validation F1-score: {results['Validation F1-Score'].max():.4f}")

# ==========================
# SAVE TEST PREDICTIONS
# ==========================
# Save predictions to CSV files
cnn_test_results.to_csv('cnn_test_predictions.csv', index=False)
svm_test_results.to_csv('svm_test_predictions.csv', index=False)
if xgb_trained:
    xgb_test_results.to_csv('xgb_test_predictions.csv', index=False)

print(f"\n=== Test Predictions Saved ===")
print("- cnn_test_predictions.csv")
print("- svm_test_predictions.csv") 
if xgb_trained:
    print("- xgb_test_predictions.csv")

# Compare model agreement on test set
agreement_data = {
    'filename': test_filenames,
    'cnn_pred': cnn_test_results['predicted_class'],
    'svm_pred': svm_test_results['predicted_class']
}

if xgb_trained:
    agreement_data['xgb_pred'] = xgb_test_results['predicted_class']

agreement_df = pd.DataFrame(agreement_data)

# Calculate agreement between models
cnn_svm_agreement = (agreement_df['cnn_pred'] == agreement_df['svm_pred']).mean()

print(f"\n=== Model Agreement on Test Set ===")
print(f"CNN-SVM agreement: {cnn_svm_agreement:.4f}")

if xgb_trained:
    cnn_xgb_agreement = (agreement_df['cnn_pred'] == agreement_df['xgb_pred']).mean()
    svm_xgb_agreement = (agreement_df['svm_pred'] == agreement_df['xgb_pred']).mean()
    all_agree = ((agreement_df['cnn_pred'] == agreement_df['svm_pred']) & 
                 (agreement_df['cnn_pred'] == agreement_df['xgb_pred'])).mean()
    
    print(f"CNN-XGBoost agreement: {cnn_xgb_agreement:.4f}")
    print(f"SVM-XGBoost agreement: {svm_xgb_agreement:.4f}")
    print(f"All models agree: {all_agree:.4f}")

# Show cases where models disagree
if xgb_trained:
    disagreement_cases = agreement_df[
        (agreement_df['cnn_pred'] != agreement_df['svm_pred']) |
        (agreement_df['cnn_pred'] != agreement_df['xgb_pred']) |
        (agreement_df['svm_pred'] != agreement_df['xgb_pred'])
    ]
else:
    disagreement_cases = agreement_df[
        agreement_df['cnn_pred'] != agreement_df['svm_pred']
    ]

if len(disagreement_cases) > 0:
    print(f"\n=== Sample Disagreement Cases ===")
    print(disagreement_cases.head(10))