import pennylane as qml
import torch
from torch import nn

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires = n_qubits)

@qml.qnode(dev, interface = "torch")

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(
    inputs,
    wires=range(n_qubits),
    rotation="Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires = range(n_qubits)
    )

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

weight_shapes = {
    "weights": (n_layers, n_qubits, 3)
}

quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)

model = nn.Sequential(
    nn.Linear(4, 4),
    nn.Tanh(),
    quantum_layer,
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_function = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

X_train = torch.tensor([
    [0.1, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.3, 0.2],
    [0.8, 0.9, 0.7, 0.9],
    [0.9, 0.8, 0.9, 0.7]
], dtype=torch.float32)

y_train = torch.tensor(
    [[0.0], [0.0], [1.0], [1.0]],
    dtype=torch.float32
)

for epoch in range(50):

    optimizer.zero_grad()

    predictions = model(X_train)

    loss = loss_function(
        predictions,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1} - Loss: {loss.item():.4f}"
    )

X_test = torch.tensor([
    [0.15, 0.25, 0.2, 0.1],
    [0.85, 0.75, 0.8, 0.9]
], dtype=torch.float32)

y_test = torch.tensor(
    [[0.0], [1.0]],
    dtype=torch.float32
)

with torch.no_grad():

    predictions = model(X_test)

    predicted_classes = (
        predictions >= 0.5
    ).float()

    accuracy = (
        predicted_classes == y_test
    ).float().mean()

    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    

