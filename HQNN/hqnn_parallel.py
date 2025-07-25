# Hybrid Quantum Neural Network (HQNN) with Parallel Quantum Circuits
# Running this Python file will simply test the components of the HQNN model.
# It will not run the full training loop or inference.


# --- Import Required Libraries ---
# These libraries are essential for building, training, and visualizing the hybrid quantum-classical neural network.
import torch
import torch.nn as nn
import pennylane as qml
import torchmetrics
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from IPython.display import display
from utils import save_model



# --- Classical Convolutional Block ---
# IF YOU GET AN ERROR WHEN SAVING / LOADING THE MODELS, TRY SWITCHING TO THE COMMENTED VERSION BELOW.
# WHEN TRAINING NEW MODELS, USE THE CLASS DEFINITION.

# def ClassicalConvBlock(in_channels=1):
#     return nn.Sequential(
#         # First Convolutional Block
#         nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
#         nn.BatchNorm2d(16),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2), # Output: (batch, 16, 14, 14)

#         # Second Convolutional Block
#         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2), # Output: (batch, 32, 7, 7)

#         # Flatten for dense layers
#         nn.Flatten() # Output: (batch, 1568)
#     )
class ClassicalConvBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (batch, 16, 14, 14)

            # Second Convolutional Block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (batch, 32, 7, 7)

            # Flatten for dense layers
            nn.Flatten() # Output: (batch, 1568)
        )
    def forward(self, x):
        return self.conv(x)

# --- Testing the ClassicalConvBlock ---
classical_conv_block = ClassicalConvBlock(in_channels=1)
# Create dummy input that models MNIST
dummy_img_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor (batch_size=1, channels=1, height=28, width=28)
output = classical_conv_block(dummy_img_tensor)
assert output.shape == (1, 1568), f"Expected output shape (1, 1568), got {output.shape}"
print("ClassicalConvBlock test passed.")

# --- ParallelQuantumLayer ---
class ParallelQuantumLayer(nn.Module):
    """
    A PyTorch module for the parallel quantum layers.
    
    This module includes a classical preprocessing layer and a set of parallel
    quantum circuits implemented with PennyLane. It's designed to be a drop-in
    replacement in a PyTorch model.
    """
    def __init__(
            self, 
            n_qubits, 
            n_circuits, 
            depth, 
            device='default.qubit', 
            shots=None, 
            plot_circuit=False
        ):
        """
        Args:
            n (int): The number of features coming from the CNN (e.g., 1568).
            n_qubits (int): The number of qubits in each quantum circuit.
            depth (int): The depth of the variational part of each quantum circuit.
            device (str): The PennyLane device to use.
            shots (int, optional): The number of shots for measurements. None for exact expectation.
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.total_quantum_features = n_circuits * n_qubits

        # --- 1. Classical Preprocessing Layer ---
        # This layer maps the flattened CNN output to the total number of features
        # required by all parallel quantum circuits.
        self.classical_preprocessor = nn.Sequential(
            nn.LazyLinear(self.total_quantum_features),
            nn.BatchNorm1d(self.total_quantum_features),
            nn.ReLU()
        )

        # --- 2. Quantum Device and Circuit Definition ---
        # We define a single quantum device and a QNode template.
        dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def quantum_circuit(inputs, weights):
            """The quantum circuit template for a single parallel layer."""
            # Reshape weights for easier indexing
            weights = weights.reshape(depth, 3, n_qubits)
            
            # Encode classical data using Angle Embedding
            # Scale inputs to [0, π] range for angle embedding
            # This ensures proper rotation angles for the quantum gates
            scaled_inputs = inputs * torch.pi
            qml.AngleEmbedding(scaled_inputs, wires=range(n_qubits), rotation='X')

            # Apply variational layers (trainable)
            for layer_id in range(depth):
                # Trainable rotation gates
                for i in range(n_qubits):
                    qml.RZ(weights[layer_id, 0, i], wires=i)
                    qml.RY(weights[layer_id, 1, i], wires=i)
                    qml.RZ(weights[layer_id, 2, i], wires=i)
                
                # Entangling gates
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

                # Add a barrier to align the layers
                qml.Barrier(wires=range(n_qubits))

            # Return expectation values for each qubit
            return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

        # --- 3. Create TorchLayers for Parallel Execution ---
        # We use qml.qnn.TorchLayer to wrap our QNode. This makes it a proper
        # PyTorch layer. We create a list of these layers, one for each parallel circuit.
        weight_shapes = {"weights": (depth * 3 * n_qubits)}
        self.quantum_layers = nn.ModuleList(
            [qml.qnn.TorchLayer(quantum_circuit, weight_shapes) for _ in range(n_circuits)]
        )

        if plot_circuit:
            # Display one of the circuits for visualization
            print("--- Quantum Circuit Structure ---")
            # Get the QNode from the first TorchLayer
            qnode_to_draw = self.quantum_layers[0].qnode
            # Create dummy inputs for drawing
            dummy_inputs = torch.zeros(self.n_qubits)
            dummy_weights = torch.zeros(weight_shapes['weights'])
            # Draw the circuit
            fig, ax = qml.draw_mpl(qnode_to_draw)(dummy_inputs, dummy_weights)
            plt.show()
            print("-------------------------------")

    def forward(self, x):
        """
        The forward pass for the parallel quantum layer.
        
        Args:
            x (torch.Tensor): Input tensor from the CNN of shape (batch_size, cnn_output_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_circuits * n_qubits).
        """
        # 1. Pass input through the classical preprocessor
        processed_features = self.classical_preprocessor(x)
        
        # 2. Split the features for each parallel quantum circuit.
        # The tensor is split along the feature dimension (dim=1).
        split_features = torch.split(processed_features, self.n_qubits, dim=1)
        
        # 3. Process each feature chunk through its corresponding quantum layer.
        quantum_outputs = [self.quantum_layers[i](split_features[i]) for i in range(self.n_circuits)]
            
        # 4. Concatenate the outputs from all quantum circuits back into a single tensor.
        return torch.cat(quantum_outputs, dim=1)
    

# --- Testing the ParallelQuantumLayer
parallel_quantum_layer = ParallelQuantumLayer(
    n_qubits=3,
    n_circuits=2,
    depth=1,
    device='lightning.qubit',
    plot_circuit=False
).eval()

# Use the output from the ClassicalConvBlock test
output = parallel_quantum_layer(output)

assert output.shape == (1, 6), f"Expected output shape (1, 6), got {output.shape}"
print("ParallelQuantumLayer test passed.")



# --- Classical Dense Layer ---
def ClassicalDenseLayer(quantum_output_size, n_classes):
    return nn.Sequential(
        nn.Linear(quantum_output_size, n_classes),
        nn.BatchNorm1d(n_classes),
    )

# --- Testing the ClassicalDenseLayer ---
classical_dense_layer = ClassicalDenseLayer(
    quantum_output_size=6, 
    n_classes=10
).eval()

output = classical_dense_layer(output)
assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
print("ClassicalDenseLayer test passed.")



# --- Full HQNNParallel Model ---
class HQNNParallel(nn.Module):
    def __init__(
            self, 
            n_circuits, 
            n_qubits, 
            depth, 
            in_channels=1, # default assume grayscale images
            n_classes=10, 
            device='lightning.qubit', 
            plot_circuit=False,
            **kwargs # ignore training hyperparameters coming in as arguments
        ):
        super().__init__()

        # 1. Classical Convolutional Block
        self.classical_conv_block = ClassicalConvBlock(
            in_channels=in_channels,
        )

        # 2. Parallel Quantum Layer
        self.parallel_quantum_layer = ParallelQuantumLayer(
            n_circuits=n_circuits,
            n_qubits=n_qubits,
            depth=depth,
            device=device,
            plot_circuit=plot_circuit
        )

        # 3. Final Classical Dense Classfication Layer
        # It's best practice to output raw scores (logits) and use
        # nn.CrossEntropyLoss, which combines Softmax and NLLLoss for
        # better numerical stability.
        quantum_output_size = n_circuits * n_qubits
        self.classical_classifier = ClassicalDenseLayer(quantum_output_size, n_classes)

    def forward(self, x):
        features = self.classical_conv_block(x)

        quantum_output = self.parallel_quantum_layer(features)

        predictions = self.classical_classifier(quantum_output)

        return predictions
    
# --- Testing the HQNNParallel ---
hqnn_parallel = HQNNParallel(
    n_circuits=2,
    n_qubits=3,
    depth=1,
    in_channels=1,
    n_classes=10,
    device='lightning.qubit',
    plot_circuit=False
).eval()

output = hqnn_parallel(dummy_img_tensor)
assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
print("HQNNParallel model test passed.")