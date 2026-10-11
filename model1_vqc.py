import pennylane as qml
from pennylane import numpy as np

n_qubits= 4
n_layers = 2

dev = qml.device("default.qubit", wires = n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(
        inputs,
        wires= range(n_qubits),
        rotation= "Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires= range(n_qubits)
    )

    return qml.expval(qml.PauliZ(0))

sample_input = np.array(
    [0.2,0.4,0.6,0.8],
    requires_grad= False
)

weight_shape = qml.StronglyEntanglingLayers.shape(
    n_layers=n_layers,
    n_wires=n_qubits

)

weights = np.random.random(
    weight_shape,
    requires_grad = True
)

output = quantum_circuit(
    sample_input,
    weights
)

print("Quantum circuit output:", output)
print("\nQuantum Circuit:")

print(qml.draw(quantum_circuit)(sample_input, weights))

optimizer = qml.GradientDescentOptimizer(stepsize= 0.1)

def cost(weights, X, y):
    predictions = [quantum_circuit(x, weights) for x in X]
    return np.mean((np.array(predictions) - y) **2)

X_train = np.array([
    [0.1, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.3, 0.2],
    [0.8, 0.9, 0.7, 0.9],
    [0.9, 0.8, 0.9, 0.7]
], requires_grad=False)

y_train = np.array(
    [-1,-1,1,1],
    requires_grad = False
)

for epoch in range(50):
    weights = optimizer.step(
        lambda w: cost(w, X_train, y_train),
        weights
    )
    current_lose = cost(weights, X_train, y_train)
    
    print(f"Epoch {epoch +1} - Lose: {current_lose}")

for x in X_train:
    prediction = quantum_circuit(x, weights)
    label = "DDos" if prediction >= 0 else "Normal"
    print(prediction, label)


X_test = np.array([
    [0.15, 0.25, 0.2, 0.1],
    [0.85, 0.75, 0.8, 0.9]
], requires_grad=False)

y_test = np.array(
    [-1, 1],
    requires_grad=False
)


correct = 0

for x, y in zip(X_test, y_test):

    prediction = quantum_circuit(
        x,
        weights
    )

    predicted_label = (
        1 if prediction >= 0 else -1
    )

    if predicted_label == y:
        correct += 1

accuracy = correct / len(X_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

