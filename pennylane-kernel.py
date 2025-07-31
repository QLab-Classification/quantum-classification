import pennylane as qml
from pennylane import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta

def load_mnist(n_qubit):
    # Load MNIST dataset from OpenML
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    print(X.shape)

    # Use 10% of the dataset for faster computation
    X = X[:10000].values
    y = y[:10000].values

    # Filter out the digits 3 and 6
    mask = (y == '3') | (y == '6')
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Convert labels to binary (0 for digit 3 and 1 for digit 6)
    y_filtered = np.where(y_filtered == '3', 0, 1)

    # Apply PCA to reduce the feature dimension
    #pca reduces the dimensionality of the dataset
    pca = PCA(n_components=n_qubit)
    X_reduced = pca.fit_transform(X_filtered)

    # Normalize the input feature
    scaler = StandardScaler().fit(X_reduced)
    X_scaled = scaler.transform(X_reduced)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

n_qubit = 8

# Load API key
def load_api_key():
    try:
        with open('api-key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("API key file not found. Please create api-key.txt with your IonQ API key.")
        return None

# Choose target device - options: "local", "simulator", "qpu.aria-1", "qpu.aria-2", "qpu.forte-1"
TARGET_DEVICE = "qpu.aria-1"  # Change this to "simulator" for IonQ or "qpu.aria-1" for actual hardware

# Create PennyLane devices
dev_local = qml.device('default.qubit', wires=n_qubit)

# Create IonQ device if API key is available
def create_ionq_device():
    if TARGET_DEVICE == "local":
        # No need to create IonQ device for local simulation
        return None
    
    api_key = load_api_key()
    if api_key:
        try:
            if TARGET_DEVICE == "simulator":
                # Use ionq.simulator device for ideal simulation
                return qml.device('ionq.simulator', wires=n_qubit, api_key=api_key, shots=1024)
            elif TARGET_DEVICE.startswith("qpu."):
                # Use ionq.qpu device for hardware (remove target parameter)
                return qml.device('ionq.qpu', wires=n_qubit, api_key=api_key, shots=1024)
            else:
                print(f"Unknown TARGET_DEVICE: {TARGET_DEVICE}. Using local simulation.")
                return None
        except Exception as e:
            print(f"Failed to create IonQ device: {e}")
            print("Falling back to local simulation.")
            return None
    else:
        print("API key not found. Using local simulation.")
        return None

dev_ionq = create_ionq_device()

@qml.qnode(dev_local, diff_method='parameter-shift')
def kernel_local(x1, x2, n_qubit):
    """Local kernel function for testing and development"""
    qml.AngleEmbedding(x1, wires=range(n_qubit))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubit))
    # Create tensor product of PauliZ operators for all qubits
    pauliz_ops = [qml.PauliZ(wires=i) for i in range(n_qubit)]
    return qml.expval(qml.prod(*pauliz_ops))

# Function to monitor IonQ job status
def monitor_ionq_job(device, max_wait_minutes=30):
    """
    Monitor IonQ job status every 30 seconds until completion or timeout
    
    Args:
        device: PennyLane IonQ device
        max_wait_minutes: Maximum time to wait in minutes
    
    Returns:
        True if job completed successfully, False otherwise
    """
    print(f"Monitoring IonQ job...")
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        try:
            # Check if the device has a job attribute (from the Job class in device.py)
            if hasattr(device, 'job') and device.job is not None:
                job = device.job
                
                # Get job ID from the job object
                if hasattr(job, 'id') and hasattr(job.id, 'value'):
                    job_id = job.id.value
                    print(f"Job ID: {job_id}")
                    
                    # Reload job to get latest status
                    job.reload()
                    
                    # Check job status
                    if job.is_complete:
                        print(f"Job {job_id} completed successfully!")
                        return True
                    elif job.is_failed:
                        print(f"Job {job_id} failed!")
                        return False
                    else:
                        print(f"Job {job_id} status: running")
                        # Wait 30 seconds before checking again
                        time.sleep(30)
                else:
                    print("No job ID found")
                    return False
            else:
                print("No job found on device")
                return False
            
        except Exception as e:
            print(f"Error monitoring job: {e}")
            return False
    
    print(f"Job monitoring timeout after {max_wait_minutes} minutes")
    return False

def kernel_ionq(x1, x2, n_qubit, shots=1024):
    """
    Kernel function that runs on IonQ hardware using PennyLane
    Returns the expectation value and prints job ID
    """
    if dev_ionq is None:
        print("IonQ device not available. Using local simulation.")
        return kernel_local(x1, x2, n_qubit)
    
    # Create quantum node with IonQ device
    @qml.qnode(dev_ionq, diff_method='finite-diff')
    def quantum_kernel(x1, x2, n_qubit):
        qml.AngleEmbedding(x1, wires=range(n_qubit))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubit))
        # Create tensor product of PauliZ operators for all qubits
        pauliz_ops = [qml.PauliZ(wires=i) for i in range(n_qubit)]
        return qml.expval(qml.prod(*pauliz_ops))
    
    try:
        # Execute the quantum function
        print(f"Executing quantum kernel on {TARGET_DEVICE}...")
        result = quantum_kernel(x1, x2, n_qubit)
        
        # For simulator, job monitoring is not always needed - just return the result
        if TARGET_DEVICE == "simulator":
            print(f"✓ Simulator execution completed successfully")
            return result
        
        # For hardware, try to get job ID and monitor
        if hasattr(dev_ionq, 'job') and dev_ionq.job is not None:
            job = dev_ionq.job
            if hasattr(job, 'id') and hasattr(job.id, 'value'):
                job_id = job.id.value
                print(f"Job submitted! Job ID: {job_id}")
                
                # Monitor job status for hardware
                job_success = monitor_ionq_job(dev_ionq, max_wait_minutes=10)
                
                if not job_success:
                    print("Job monitoring failed. Using local simulation.")
                    return kernel_local(x1, x2, n_qubit)
                
                # If we reach here, job was successful
                return result
            else:
                print("No job ID found on device. Using local simulation.")
                return kernel_local(x1, x2, n_qubit)
        else:
            print("No job found on device. Using local simulation.")
            return kernel_local(x1, x2, n_qubit)
        
    except Exception as e:
        print(f"Error executing on IonQ device: {e}")
        print("Falling back to local simulation.")
        return kernel_local(x1, x2, n_qubit)

# Choose which kernel to use
def kernel(x1, x2, n_qubit):
    """Main kernel function - switches between local and IonQ based on configuration"""
    if TARGET_DEVICE == "local":
        # Use local simulation
        return kernel_local(x1, x2, n_qubit)
    elif TARGET_DEVICE == "simulator" and dev_ionq is not None:
        # Use IonQ simulator
        return kernel_ionq(x1, x2, n_qubit, shots=100)  # Fewer shots for simulator
    elif TARGET_DEVICE.startswith("qpu.") and dev_ionq is not None:
        # Use IonQ hardware
        return kernel_ionq(x1, x2, n_qubit, shots=1024)
    else:
        # Default to local simulation
        return kernel_local(x1, x2, n_qubit)
    # samples = qml.sample(qml.PauliZ(wires=n_qubit))
    # # generate counts dict
    # counts = {}
    # for sample in samples:
    #     key = ''.join(map(str, sample))
    #     counts[key] = counts.get(key, 0) + 1
    # # Convert counts to probabilities
    # total_counts = sum(counts.values())
    # for key in counts:
    #     counts[key] /= total_counts

    # # Use probabilities to calculate expectation value
    # expectation = 0
    # for key, prob in counts.items():
    #     parity = (-1) ** key.count('1')
    #     expectation += parity * prob
    # expectation = expectation / n_qubit

    # return expectation


def kernel_mat(A, B, desc="Computing kernel matrix"):
    """Compute kernel matrix with progress tracking"""
    total_calculations = len(A) * len(B)
    current_calculation = 0
    start_time = time.time()
    
    print(f"\n{desc}")
    print(f"Total calculations needed: {total_calculations}")
    print(f"Starting at: {datetime.now().strftime('%H:%M:%S')}")
    
    mat = []
    for i, a in enumerate(A):
        row = []
        for j, b in enumerate(B):
            current_calculation += 1
            
            # Calculate kernel value
            kernel_value = kernel(a, b, n_qubit)
            row.append(kernel_value)
            
            # Progress tracking
            elapsed_time = time.time() - start_time
            avg_time_per_calc = elapsed_time / current_calculation
            remaining_calcs = total_calculations - current_calculation
            estimated_remaining = remaining_calcs * avg_time_per_calc
            
            # Print progress every 5 calculations or at the end
            if current_calculation % 5 == 0 or current_calculation == total_calculations:
                progress_pct = (current_calculation / total_calculations) * 100
                eta = datetime.now() + timedelta(seconds=estimated_remaining)
                print(f"Progress: {current_calculation}/{total_calculations} ({progress_pct:.1f}%) | "
                      f"Avg time: {avg_time_per_calc:.1f}s | "
                      f"ETA: {eta.strftime('%H:%M:%S')}")
        
        mat.append(row)
    
    total_time = time.time() - start_time
    print(f"✓ Completed in {total_time:.1f}s (avg: {total_time/total_calculations:.1f}s per calculation)\n")
    
    return np.array(mat)

# Load data for both approaches
X_train, X_test, y_train, y_test = load_mnist(n_qubit)

# Full dataset evaluation (commented out for performance - use reduced sample below)
# svm = SVC(kernel=kernel_mat)
# svm.fit(X_train, y_train)
# pred = svm.predict(X_test)
# accuracy_score(y_test, pred)

# Use reduced sample for practical testing
svm = SVC(kernel='precomputed')
n_sample_max = 2  # Reduced to 2 for very fast testing (~10 calculations total)
X_train_sample = []
y_train_sample = []
for label in np.unique(y_train):
    index = y_train == label
    X_train_sample.append(X_train[index][:n_sample_max])
    y_train_sample.append(y_train[index][:n_sample_max])
X_train_sample = np.concatenate(X_train_sample, axis=0)
y_train_sample = np.concatenate(y_train_sample, axis=0)

print(f"\n=== TRAINING PHASE ===")
print(f"Training samples shape: {X_train_sample.shape}")
print(f"Computing training kernel matrix ({len(X_train_sample)}x{len(X_train_sample)})")
kernel_mat_train = kernel_mat(X_train_sample, X_train_sample, "Training kernel matrix")

print(f"\n=== TEST PHASE ===")
print(f"Test samples shape: {X_test.shape}")
# Use only first 3 test samples for very fast testing
X_test_small = X_test[:3]
y_test_small = y_test[:3]
print(f"Using smaller test set: {X_test_small.shape}")
print(f"Computing test kernel matrix ({len(X_test_small)}x{len(X_train_sample)})")
kernel_mat_test = kernel_mat(X_test_small, X_train_sample, "Test kernel matrix")

print('reached here')

print(f"\n=== ACCURACY EVALUATION ===")
accuracy = []
n_samples = []
total_iterations = len(range(2, n_sample_max+1, 1))

for iteration, n_sample in enumerate(range(2, n_sample_max+1, 1), 1):  # Test with 2,3,4,5 samples
    print(f'\n--- Iteration {iteration}/{total_iterations}: {n_sample} samples per class ---')
    
    class1_indices = np.arange(n_sample)
    class2_indices = np.arange(n_sample_max, n_sample_max+n_sample)
    selected_indices = np.concatenate([class1_indices, class2_indices])

    # Train SVM
    print("Training SVM...")
    svm.fit(kernel_mat_train[np.ix_(selected_indices, selected_indices)], 
            np.concatenate([y_train_sample[:n_sample], y_train_sample[n_sample_max:n_sample_max+n_sample]]))
    
    # Make predictions
    print("Making predictions...")
    pred = svm.predict(np.concatenate([kernel_mat_test[:, :n_sample], kernel_mat_test[:, n_sample_max:n_sample_max+n_sample]], axis=1))
    
    # Calculate accuracy
    acc = accuracy_score(y_test_small, pred)
    accuracy.append(acc)
    n_samples.append(n_sample)
    print(f"✓ Accuracy: {acc:.4f}")

print(f"\n=== FINAL RESULTS ===")
for i, (n_samp, acc) in enumerate(zip(n_samples, accuracy)):
    print(f"Samples per class: {n_samp} | Accuracy: {acc:.4f}")

print(f"\n=== PLOTTING RESULTS ===")

plt.plot(n_samples, accuracy, marker='o')
plt.title('Classification Accuracy vs. #Training Samples')
plt.xlabel('#Training Samples')
plt.xticks(n_samples, n_samples)
plt.ylabel('Accuracy')
plt.grid()
plt.tight_layout()
plt.show()