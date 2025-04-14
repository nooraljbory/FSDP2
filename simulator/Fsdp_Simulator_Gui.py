import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import tempfile

st.set_page_config(page_title="FSDP Simulator", layout="wide")

menu = st.sidebar.radio("Navigation", ["🧠 Simulator", "📊 Evaluation"])

if menu == "🧠 Simulator":
    st.title("🧐 FSDP Interactive Simulator")
    st.markdown("### Fully Sharded Data Parallel (FSDP) Simulator")

    # --- Model Configuration ---
    st.sidebar.header("Model Configuration")
    hidden_layers = st.sidebar.number_input("Number of Hidden Layers", min_value=1, value=2)

    neuron_input = st.sidebar.text_input("Neurons per Layer (comma separated)", value="64,32")
    neurons_per_layer = [int(n) for n in neuron_input.split(",") if n.strip().isdigit()]
    if len(neurons_per_layer) != hidden_layers:
        st.sidebar.error("Number of neurons must match the number of hidden layers")

    activation_choices = ["ReLU", "Sigmoid", "Tanh", "Softmax"]
    activation_selected = st.sidebar.multiselect("Activation Functions", activation_choices, default=["ReLU"])
    activation_map = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1)
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

    # --- Generate and Show Dataset Info ---
    num_records = st.number_input("Number of Records", min_value=10, value=100)
    num_features = st.number_input("Number of Features", min_value=2, value=10)
    X = torch.randn(num_records, num_features)
    y = torch.randint(0, 2, (num_records,))

    st.write(f"Dataset contains: **{num_records} records**, **{num_features} features**")

    train_percentage = st.slider("Training data %", 10, 100, 80)
    split_idx = int(num_records * train_percentage / 100)
    train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
    test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- Model Definition ---
    class SimpleModel(nn.Module):
        def __init__(self, layers, activations):
            super().__init__()
            modules = []
            for i in range(len(layers) - 1):
                modules.append(nn.Linear(layers[i], layers[i + 1]))
                if i < len(activations):
                    modules.append(activations[i])
            self.net = nn.Sequential(*modules)

        def forward(self, x):
            return self.net(x)

    if neurons_per_layer:
        model = SimpleModel([X.shape[1]] + neurons_per_layer + [2], activation_functions)
        loss_map = {
            "crossentropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "bce": nn.BCELoss()
        }
        loss_fn = loss_map.get(loss_function_name.lower(), nn.CrossEntropyLoss())
        optimizer_cls = getattr(optim, optimizer_name)
        optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

        # --- Training Loop ---
        if st.button("🚀 Start Training"):
            st.write("Training in progress...")
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_x)
                    try:
                        loss = loss_fn(output, batch_y)
                    except Exception as e:
                        st.error(f"Loss calculation failed: {e}")
                        st.stop()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                st.write(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

            st.success("Training complete!")

            # --- Evaluation ---
            model.eval()
            y_true, y_pred, y_score = [], [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    output = model(batch_x)
                    pred = output.argmax(dim=1)
                    y_true.extend(batch_y.tolist())
                    y_pred.extend(pred.tolist())
                    probs = torch.softmax(output, dim=1)[:, 1]
                    y_score.extend(probs.tolist())

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score)

            st.session_state['evaluation'] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_score": y_score,
                "metrics": {
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "AUC": auc
                }
            }

if menu == "📊 Evaluation" and 'evaluation' in st.session_state:
    st.title("📊 Evaluation Metrics")
    metrics = st.session_state['evaluation']['metrics']
    st.json(metrics)

    fpr, tpr, _ = roc_curve(st.session_state['evaluation']['y_true'], st.session_state['evaluation']['y_score'])
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
        "metrics": metrics
    }
    report_filename = f"evaluation_report_{now}.json"
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmpfile:
        json.dump(report_data, tmpfile, indent=4)
        tmpfile.flush()
        with open(tmpfile.name, "rb") as f:
            st.download_button("📥 Download Evaluation Report", f, file_name=report_filename, mime="application/json")
