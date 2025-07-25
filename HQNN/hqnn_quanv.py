"""
HQNN-Quanv Model Implementation
Source: HQNN-Quanv-local.ipynb

This file contains the QuanvolutionalLayer, HQNNQuanv model, training loop, model loading, and prediction utilities.
All code is annotated with verbose, beginner-friendly comments and docstrings, including explanations from the original notebook.
"""

# --- Imports ---
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

class QuanvolutionalLayer(nn.Module):
    '''
    A PyTorch module for the quanvolutional layer.
    '''

    def __init__(self, kernel_size=4, depth=1, draw_circuit=False):
        '''
        Args:
            kernel_size (int): The size of the kernel. e.g. passing 2 would result in a 2x2 kernel.
            depth (int): The number of repetitions of variational sections to use. 
                - This linearly affects the number of trainable parameters in the quanvolutional layer.
            device (str): The PennyLane device to use.
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.n_qubits = kernel_size**2
        self.depth = depth
        
        self.draw_circuit = draw_circuit

        # Learnable weights for the variational circuit
        self.weights = nn.Parameter(torch.randn(self.depth, self.n_qubits))

        
        # --- Creating the PennyLane device to run circuits on ---
        dev = qml.device('lightning.qubit', wires=self.n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def quantum_circuit(inputs, weights):
            # --- Encoding the inputs into Y-rotation angles ---
            # Scale inputs to [0, pi] range
            scaled_inputs = inputs * torch.pi
            qml.AngleEmbedding(scaled_inputs, wires=range(self.n_qubits), rotation='Y')

            # --- Define the Variational layers (trainable params) ---
            # Reshape the weights for easier indexing
            weights = weights.reshape(self.depth, self.n_qubits)
            for layer_id in range(self.depth):
                # --- TODO: Support other kernel sizes ---
                # X gates with trainable parameters on qubits 0 and 1
                qml.RX(weights[layer_id, 0], wires=0)
                qml.RX(weights[layer_id, 1], wires=1)
                # CNOT gates
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[0, 2])
                qml.CNOT(wires=[0, 3])
                # Y gates with trainable parameters on qubits 0 and 3
                qml.RY(weights[layer_id, 2], wires=0)
                qml.RY(weights[layer_id, 3], wires=3)

                # separate the layers with a barrier
                qml.Barrier(wires=range(self.n_qubits))

            # --- Measurement of the qubits ---
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.quantum_kernel = quantum_circuit

        if self.draw_circuit:
            # --- Display the circuit for visualization ---
            print("--- Quantum Circuit Structure ---")
            # Create dummy inputs and weights for drawing
            dummy_inputs = torch.zeros(self.n_qubits)
            dummy_weights = torch.zeros(self.depth * self.n_qubits)

            # Draw the circuit
            fig, ax = qml.draw_mpl(quantum_circuit)(dummy_inputs, dummy_weights)
            plt.show()
            print("---------------------------------")

    def forward(self, x):
        '''
        Forward pass for the quanvolutional layer.
        '''

        # assume x is a batch of patches of shape (batch, n_qubits)
        batch_size = x.shape[0]
        weights = torch.zeros(self.depth * self.n_qubits, device=x.device, dtype=x.dtype)

        outputs = []
        for i in range(batch_size):
            out = torch.tensor(self.quantum_kernel(x[i], weights), device=x.device, dtype=x.dtype)
            outputs.append(out)

        return torch.stack(outputs)
    
# --- Testing the Quanvolutional Layer ---
quanv_layer = QuanvolutionalLayer(
    kernel_size=2,
    depth=1,
    draw_circuit=False
)
dummy_input = torch.randn(3, 4)
output = quanv_layer(dummy_input)
assert output.shape == (3, 4), "Output shape mismatch: expected (3, 4), got {}".format(output.shape)
print("Quanvolutional Layer test passed.")



# --- HQNNQuanv Model Definition ---
class HQNNQuanv(nn.Module):
    def __init__(
        self, 
        image_size=8, 
        kernel_size=2, 
        depth=1, 
        in_channels=1, 
        n_classes=10, 
        draw_circuit=False, 
        **kwargs
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.depth = depth

        # Quanvolutional layer
        self.quanv = QuanvolutionalLayer(kernel_size=kernel_size, depth=depth, draw_circuit=draw_circuit)

        # Calculate output size after sliding the kernel
        # For stride=1, padding=0:
        self.feature_map_size = image_size - kernel_size + 1
        self.n_channels = self.n_qubits * in_channels # 4 for 2x2 kernel

        # Fully connected layer
        self.Dense = nn.Linear(self.n_channels * self.feature_map_size**2, n_classes)

    def forward(self, x):
        # x: (batch, 1, H, W)
        batch_size = x.shape[0]
        patches = []
        # Extract patches and apply quanvolution
        for img in x:
            # img: (1, H, W)
            img_patches = []
            for c in range(img.shape[0]):
                for i in range(self.feature_map_size):
                    for j in range(self.feature_map_size):
                        patch = img[c, i:i+self.kernel_size, j:j+self.kernel_size].reshape(-1)
                        img_patches.append(patch)
            
            img_patches = torch.stack(img_patches)      # (n_patches, n_qubits)
            quanv_out = self.quanv(img_patches)         # (n_patches, n_qubits)
            quanv_out = quanv_out.transpose(0, 1)       # (n_qubits, n_patches)
            quanv_out = quanv_out.reshape(self.n_channels, self.feature_map_size, self.feature_map_size)
            patches.append(quanv_out)

        x = torch.stack(patches)    # (batch_size, n_channels, feature_map_size, feature_map_size)

        x = x.view(batch_size, -1)  # Flatten

        x = self.Dense(x) # Fully connected layer

        return x
    

# --- Testing the HQNNQuanv Model ---
model = HQNNQuanv(
    image_size=28,
    kernel_size=2,
    depth=1,
    n_classes=10,
    draw_circuit=False
)
dummy_images = torch.randn(
    2,       # batch_size
    1,      # input_channels
    28,     # image_width
    28      # image_height
)
output = model(dummy_images)
assert output.shape == (2, 10), "Output shape mismatch: expected (2, 10), got {}".format(output.shape)
print("HQNNQuanv Model test passed.")
