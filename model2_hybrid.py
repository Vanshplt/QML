import pennylane as qml
import torch
from torch import nn

#Using 4 qubits and 2 trainable layers
n_qubits = 4
n_layers = 2

#This creates the quantum computer to run our circuits
dev = qml.device("default.qubit", wires = n_qubits)

#converts function below into pennylane quantum circuit
#interface = "torch" allows pennylane to work with pytorch
@qml.qnode(dev, interface = "torch")

#this defines the quantum circuit function which will accept inputs and weights
#input is the data and weights is the parameter model learns during training
def quantum_circuit(inputs, weights):
    #The angleEmbedding encodes the classical data into quantum circuit.
    qml.AngleEmbedding(
    inputs,
    wires=range(n_qubits),
    #values are used as angles for rotation around the Y-axis of Bloch sphere
    rotation="Y"
    )

    #this is the learnable part of the quantum model
    qml.StronglyEntanglingLayers(
        #control quantum rotation gates
        weights,
        wires = range(n_qubits)
    )

    #measures every qubit using Pauli-Z observable 
    #unlike model 1 where we only measureed one qubit, here we get measurement
    # for all 4 of them.
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

#this tells pennylane the shape of trainable quantum weights. we have
# 2 quantum layers, 4 qubits and 3 rotation parameter per qubit
weight_shapes = {
    "weights": (n_layers, n_qubits, 3)
}

#Converts the pennylane quantum circuit into Pytorch layer which allows us
#to place our quantum circuit inside normal pytorch neural network.
quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)

#nn.Sequential passes the data through each layer in order from top to bottom
model = nn.Sequential(
    # classical neaural network layer takes 4 input features and give 4 output values
    nn.Linear(4, 4),
    #adds non-linearity to classical part of the model
    nn.Tanh(),
    #the 4 layers from the classical layer are encoded into 4 qubit quantum circuit
    quantum_layer,
    #the quantum circuit produces 4 measurements. this layer converts those 4 values
    #into 1 value
    nn.Linear(4, 1),
    #sigmoid will convert the final value into number between 0 and 1
    #in our case 0 is normal traffic and 1 is DDos traffic
    nn.Sigmoid()
)

#this creates the loss function. nn.BCELoss means binary cross entropy loss.
#it is suitable because this is a binary classification problem with 0 and 1
loss_function = nn.BCELoss()

#this creates the adam optimizer which changes the models trainable parameters
#to try and reduce the loss. 
optimizer = torch.optim.Adam(
    #model.parametrs includes both classical and the quantum parameters 
    model.parameters(),
    #this it the learning rate
    lr=0.01
)

#this is our temp training data. there are 4 training examples
#each example contains 4 numerical features.
X_train = torch.tensor([
    [0.1, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.3, 0.2],
    [0.8, 0.9, 0.7, 0.9],
    [0.9, 0.8, 0.9, 0.7]
], dtype=torch.float32)

#this is the correct lables where 0 is normal and 1 is DDos
y_train = torch.tensor(
    [[0.0], [0.0], [1.0], [1.0]],
    dtype=torch.float32
)

#we are training the model on 50 epochs
for epoch in range(50):
    #pytorch accumulates gradient by default
    #so we need to reset it before each training
    optimizer.zero_grad()
    #passes all training examples through the hybrid model which produces
    #one prediction for each training example
    predictions = model(X_train)
    #compares the models prediction to the correct labels
    loss = loss_function(
        predictions,
        y_train
    )
    #calculates the gradients which tell us how each trainable parameter affects the loss
    #gradients are calculated in both classical and quantum layer
    loss.backward()
    #updates the model parameter using the gradients. Adam tries to change
    #the parameters in a direction that reduces the loss
    optimizer.step()
    print(
        f"Epoch {epoch + 1} - Loss: {loss.item():.4f}"
    )

#creating seperate test data to see how the model performs on this data 
#on the one it was not trained
X_test = torch.tensor([
    [0.15, 0.25, 0.2, 0.1],
    [0.85, 0.75, 0.8, 0.9]
], dtype=torch.float32)

#correct labels for the test data
y_test = torch.tensor(
    [[0.0], [1.0]],
    dtype=torch.float32
)

#torch.no_grad tells pytorch that gradients are not needed during this section 
#which makes the testing more efficient
with torch.no_grad():
    #run the unseen test examples through the trained model
    predictions = model(X_test)
    #converts model output probabilities into classes.
    #less than 0.5 is normal and greater than 0.5 is ddos
    predicted_classes = (
        predictions >= 0.5
    ).float()
    #compares the predicted classes with correct test labels
    accuracy = (
        predicted_classes == y_test
    ).float().mean()

    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")



