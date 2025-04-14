import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import tempfile

st.set_page_config(page_title="FSDP Simulator", layout="wide")

st.title("🧐 FSDP Interactive Simulator")
st.markdown("### Fully Sharded Data Parallel (FSDP) Simulator")

# --- Model Configuration ---
st.sidebar.header("Model Configuration")
hidden_layers = st.sidebar.number_input("Number of Hidden Layers", min_value=1, value=2)

neuron_input = st.sidebar.text_input("Neurons per Layer (comma separated)", value="64,32")
neurons_per_layer = [int(n) for n in neuron_input.split(",") if n.strip().isdigit()]
if len(neurons_per_layer) != hidden_layers:
    st.sidebar.error("Number of neurons must match the number of hidden layers")

activation_choices = ["ReLU", "Sigmoid", "Tanh"]
activation_selected = st.sidebar.multiselect("Activation Functions", activation_choices, default=["ReLU"])
activation_map = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}
activation_functions = []
for a in activation_selected:
    a_lower = a.lower()
    if a_lower in activation_map:
        activation_functions.append(activation_map[a_lower])
    else:
        st.sidebar.warning(f"Unsupported activation function: {a}")

layer_type = st.sidebar.selectbox("Layer Type", ["Dense", "Conv", "LSTM"])
deep_learning_algo = st.sidebar.selectbox("Deep Learning Algorithm", ["Autoencoder", "Classifier"])
epochs = st.sidebar.slider("Epochs", 1, 100, 10)

# --- Training Parameters ---
st.sidebar.header("Training Parameters")
optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%f")
batch_size = st.sidebar.number_input("Batch Size", value=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.5)
loss_function_name = st.sidebar.selectbox("Loss Function", ["CrossEntropy", "MSE", "BCE"])
sharding_type = st.sidebar.selectbox("Sharding Type", ["Full Sharded", "Sharded Gradient Optimizer", "No Sharding"])
gpu_count = st.sidebar.selectbox("Number of GPUs", list(range(1, 9)), index=0)

# --- Dataset Creation ---
st.sidebar.header("Dataset Info")
num_records = st.sidebar.number_input("Number of Records", min_value=10, value=100)
num_features = st.sidebar.number_input("Number of Features", min_value=2, value=10)

X_np, y_np = make_classification(n_samples=num_records, n_features=num_features, n_classes=2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)

st.write(f"Dataset contains: **{num_records} records**, **{num_features} features**")

train_percentage = st.slider("Training data %", 10, 100, 80)
split_idx = int(num_records * train_percentage / 100)
train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --- Model Definition ---
class SimpleModel(nn.Module):
    def __init__(self, layers, activations, dropout):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(activations):
                modules.append(activations[i])
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

if neurons_per_layer:
    output_dim = 1 if loss_function_name in ["MSE", "BCE"] else 2
    if output_dim == 1 and "sigmoid" not in [a.lower() for a in activation_selected]:
        activation_functions.append(nn.Sigmoid())

    model = SimpleModel([X.shape[1]] + neurons_per_layer + [output_dim], activation_functions, dropout_rate)
    loss_map = {
        "crossentropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "bce": nn.BCELoss()
    }
    loss_fn = loss_map.get(loss_function_name.lower(), nn.CrossEntropyLoss())
    optimizer_cls = getattr(optim, optimizer_name)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    if st.button("🚀 Start Training"):
        st.write("Training in progress...")
        model.train()
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                try:
                    if output_dim == 1:
                        batch_y = batch_y.unsqueeze(1).float()
                    loss = loss_fn(output, batch_y)
                except Exception as e:
                    st.error(f"Loss calculation failed: {e}")
                    st.stop()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_list.append(total_loss)
            st.write(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        st.success("Training complete!")

        # --- Loss Curve ---
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(range(1, epochs + 1), loss_list, marker='o', color='blue')
        ax_loss.set_title("Training Loss Curve")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)
        st.pyplot(fig_loss)

        # --- Evaluation ---
        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                if output_dim == 1:
                    probs = torch.sigmoid(output).squeeze()
                    pred = (probs > 0.5).long()
                else:
                    probs = torch.softmax(output, dim=1)[:, 1]
                    pred = output.argmax(dim=1)
                y_true.extend(batch_y.tolist())
                y_pred.extend(pred.tolist())
                y_score.extend(probs.tolist())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)

        st.json({
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.grid(True)
        st.pyplot(fig)

        # --- Save Report ---
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_data = {
            "metrics": {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc
            }
        }
        report_filename = f"evaluation_report_{now}.json"
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmpfile:
            json.dump(report_data, tmpfile, indent=4)
            tmpfile.flush()
            with open(tmpfile.name, "rb") as f:
                st.download_button("📥 Download Evaluation Report", f, file_name=report_filename, mime="application/json")
