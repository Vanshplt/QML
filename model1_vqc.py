#importing pennylane
import pennylane as qml
from pennylane import numpy as np

#for now this is just a python variables where we have 4 quantum circuit and
#we have 2 layers of trainable quantum portions
n_qubits= 4
n_layers = 2

#This creates the quantum computer to run our circuits
dev = qml.device("default.qubit", wires = n_qubits)

#converts function below into pennylane quantum circuit
@qml.qnode(dev)

#this defines the quantum circuit function which will accept inputs and weights
#input is the data and weights is the parameter model learns during training
def quantum_circuit(inputs, weights):

    #The angleEmbedding encodes the classical data into quantum circuit.
    qml.AngleEmbedding(
        inputs,
        wires= range(n_qubits),
        #values are used as angles for rotation around the Y-axis of Bloch sphere
        rotation= "Y"
    )

    # this is the learnable part of the quantum model
    qml.StronglyEntanglingLayers(
        #control quantum rotation gates
        weights,
        wires= range(n_qubits)
    )

    #returns the expectation value by measuring Qubit 0 using the qml.PuliZ(0)
    return qml.expval(qml.PauliZ(0))

# Creates one example with 4 features 
sample_input = np.array(
    [0.2,0.4,0.6,0.8],
    #by using false we let the model know not to modify this values
    #as they are the inputs not weights
    requires_grad= False
)

# the StronglyEntanglingLayers operation expects weifhts in a specific shape
#Pennylane calculates what shape our weights have.
weight_shape = qml.StronglyEntanglingLayers.shape(
    n_layers=n_layers,
    n_wires=n_qubits

)

#Creates random initial value for trainable weights
weights = np.random.random(
    weight_shape,
    #the optimizer gradually changes them as this time as we have
    #required_grad = True as we want the variables to be trainable.
    requires_grad = True
)

#this will run the circuit once and send the input and the weights 
#to quantum_circuit and the store the resulting mesurements in output
output = quantum_circuit(
    sample_input,
    weights
)

print("Quantum circuit output:", output)
print("\nQuantum Circuit:")

#draws text representation of the circuit
print(qml.draw(quantum_circuit)(sample_input, weights))

#optimizer changes weights so the model improves.
# stepsize= 0.1 - controls how large each update is
optimizer = qml.GradientDescentOptimizer(stepsize= 0.1)

#this tells us how badly the model is performing
def cost(weights, X, y):
    # for every example x inside X run it through quantum circuit using the current weights
    predictions = [quantum_circuit(x, weights) for x in X]
    #calculates the mean squared error.
    #it takes the predictionand the correct label then finds the difference which is then
    #squared. this is done for all predictions and then it takes their average
    #hence small cost is predictions are closer to correct answers
    #and large cost is prediction is farther from the answers.
    return np.mean((np.array(predictions) - y) **2)

#this is toy training data. we have 4 training examples where each example contains
#4 features as we have 4 qubits.
X_train = np.array([
    [0.1, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.3, 0.2],
    [0.8, 0.9, 0.7, 0.9],
    [0.9, 0.8, 0.9, 0.7]
], requires_grad=False)

# this is the correct labels where -1 is normal and 1 is DDos
y_train = np.array(
    [-1,-1,1,1],
    requires_grad = False
)

#this means we are training the model 50 times. I found that the more epochs I used the 
#more correct answer I got.
for epoch in range(50):
    # the optimizer looks at cost(w, X_train, y_train) and figures out which direction
    #should it change the weights to decrease the cost. it then replaces it with imporved weights
    weights = optimizer.step(
        lambda w: cost(w, X_train, y_train),
        weights
    )
    #this calculates the cost after latest update
    current_lose = cost(weights, X_train, y_train)
    
    print(f"Epoch {epoch +1} - Lose: {current_lose}")

#after training for x in X_train loops through the training samples again.
for x in X_train:
    #this runs each sample through the trained circuit.
    prediction = quantum_circuit(x, weights)
    # this conditional expression means if prediction is greater than or equal to 0 then it is 
    #DDos if its less then 0 then it is normal traffic.
    label = "DDos" if prediction >= 0 else "Normal"
    print(prediction, label)

#this are examples that were not used to update the weights as we want to know if our model 
# makes predictions on data it didn't directly train on. 

X_test = np.array([
    [0.15, 0.25, 0.2, 0.1],
    [0.85, 0.75, 0.8, 0.9]
], requires_grad=False)

# this is the answer for the Above test
y_test = np.array(
    [-1, 1],
    requires_grad=False
)

#increases the counter each time correct prediction is made
correct = 0

#the zip pairs test input and correct label together so iteration 1 might be x = [0.15, 0.25, 0.2, 0.1]
# y = -1 and iteration 2 will be x = [0.85, 0.75, 0.8, 0.9] y = 1
for x, y in zip(X_test, y_test):

    #gets the raw quantum prediction
    prediction = quantum_circuit(
        x,
        weights
    )

    #turns that continuous number into numerical class
    predicted_label = (
        1 if prediction >= 0 else -1
    )

    #checks if the models prediciton match the actual answers
    if predicted_label == y:
        correct += 1

# This calculates accuracy so if both correct its 100% if one is correct 50% if none then 0%
accuracy = correct / len(X_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

