# --- Imports ---
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from torch.utils.data import DataLoader, random_split
import torchmetrics
import torch.nn as nn
from IPython.display import display
from torchvision import datasets
import kagglehub
from tqdm import tqdm
import random

# --- Load MedicalMNIST dataset ---
def load_medical_mnist(transform=None, train_test_split=0.8, dataset_portion=1.00):
    dataset_path = kagglehub.dataset_download("andrewmvd/medical-mnist")
    print("Path to Medical MNIST dataset files: ", dataset_path)

    # Load the dataset using torchvision's ImageFolder
    medical_mnist = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"Total samples in Medical MNIST dataset: {len(medical_mnist)}")

    # Split into train and test
    train_dataset, test_dataset = random_split(medical_mnist, [train_test_split, 1-train_test_split])

    # Get specified portion of the dataset
    train_dataset = torch.utils.data.Subset(train_dataset, range(int(len(train_dataset) * dataset_portion)))
    test_dataset = torch.utils.data.Subset(test_dataset, range(int(len(test_dataset) * dataset_portion)))
    print(f"Using {len(train_dataset)} samples for training and {len(test_dataset)} samples for testing.")

    return train_dataset, test_dataset

# --- Load MNIST Dataset ---
def load_mnist(transform=None, dataset_portion=1.00):
    # Get train and test datasets
    train_dataset = datasets.MNIST(root='./data/mnist-parallel/train', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data/mnist-parallel/test', train=False, download=True, transform=transform)

    # Get specified portions of the dataset
    train_dataset = torch.utils.data.Subset(train_dataset, range(int(len(train_dataset) * dataset_portion)))
    test_dataset = torch.utils.data.Subset(test_dataset, range(int(len(test_dataset) * dataset_portion)))
    print(f"Using {len(train_dataset)} samples for training and {len(test_dataset)} samples for testing.")

    return train_dataset, test_dataset

# --- Training Loop ---
def train_model(
        model,
        train_dataset, 
        test_dataset, 
        n_classes,
        epochs=5, 
        batch_size=32, 
        learning_rate=0.01, 
        plot=True,
        print_every_step=False,
        **kwargs
    ):
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Define accuracy metric
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

    steps_per_epoch = len(train_loader)

    # Calculate log interval to print mid-epoch
    updates_per_epoch = steps_per_epoch if print_every_step else 12  # Number of updates to log per epoch
    log_interval = max(1, steps_per_epoch // updates_per_epoch)

    
    # --- Live Plotting Setup (Optional) ---
    if plot:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx() # Create a second y-axis sharing the same x-axis

        # Initialize plot elements
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='b')
        ax2.set_ylabel("Accuracy (%)", color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Create empty line objects to update later
        line1, = ax1.plot([], [], 'b-', label='Training Loss')
        line2, = ax2.plot([], [], 'r.-', label='Test Accuracy')
        
        # Add a legend
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        
        # Display the initial plot
        plot_display = display(fig, display_id=True)
    
    # Lists to store metrics
    train_losses, train_steps = [], []
    test_accuracies, test_epochs = [], []
    history = {
        'train_losses': train_losses,
        'train_steps': train_steps,
        'test_accuracies': test_accuracies,
        'test_epochs': test_epochs
    }
    global_step = 0

    try:
        print("Starting training...")
        for epoch in range(epochs):
            model.train() # Set model to training mode
            for step, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad() # Reset gradients

                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_steps.append(epoch + step / steps_per_epoch)
                global_step += 1

                # Log progress and update plot
                if (step + 1) % log_interval == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    if plot:
                        line1.set_data(train_steps, train_losses)
                        ax1.relim()
                        ax1.autoscale_view()
                        plot_display.update(fig)
        
            # --- Evaluation Phase ---
            model.eval()
            accuracy_metric.reset() # Reset metric for the new epoch
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    accuracy_metric.update(outputs, labels)

            accuracy = accuracy_metric.compute().item() * 100
            test_accuracies.append(accuracy)
            test_epochs.append(epoch + 1)
            print(f'Epoch [{epoch+1}/{epochs}] - Test Accuracy: {accuracy:.2f}%')

            # Update the live plot
            if plot:
                line2.set_data(test_epochs, test_accuracies)
                ax2.relim()
                ax2.autoscale_view()
                plot_display.update(fig)


        print('\n\nFinished Training')
    except KeyboardInterrupt:
        print("Training interrupted by user. Returning partial history...")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        return history


# --- Model Saving Utility ---
def save_model(model, hyperparams, history, filename):
    """
    Saves the trained HQNNQuanv model checkpoint to disk.
    Includes model state, hyperparameters, and training history.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = filename
    ext = '.pth'
    directory = f"models/{base}"
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{base}_{timestamp}{ext}"
    save_dict = {
        'state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'history': history
    }
    torch.save(save_dict, filename)
    print(f"Model saved to {filename}")
    return filename

# --- Model Loading Utility ---
def load_model(ModelClass, filepath):
    """
    Loads a saved HQNNQuanv model checkpoint from disk and restores its state.
    Also plots the training loss and test accuracy curves from the saved history.
    """
    # --- Load checkpoint from disk ---
    checkpoint = torch.load(filepath)
    hyperparams = checkpoint['hyperparams']
    state_dict = checkpoint['state_dict']
    history = checkpoint['history']

    # --- Recreate model with loaded hyperparameters ---
    loaded_model = ModelClass(**hyperparams)

    # --- Restore model weights ---
    loaded_model.load_state_dict(state_dict)

    # --- Plot training loss and test accuracy ---
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='b')
    ax2.set_ylabel("Accuracy (%)", color='r')
    ax1.plot(history["train_steps"], history["train_losses"], 'b-', label='Training Loss')
    ax2.plot(history["test_epochs"], history["test_accuracies"], 'r.-', label='Test Accuracy')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

    return loaded_model

# --- Prediction Utility ---
def predict_image(model, dataset):
    """
    Runs inference on a single image (or batch) and prints class probabilities.
    Applies softmax to output logits to get probabilities.
    Prints probability for each class and returns predicted class and confidence score.
    """
    # --- Get a random image ---
    image_tensor, actual_label = random.choice(dataset)

    # --- Display the Image with Caption ---
    img_np = image_tensor[0].squeeze().cpu().numpy()
    class_names = dataset.dataset.dataset.classes
    actual_class_name = class_names[actual_label] if actual_label < len(class_names) else str(actual_label)

    plt.figure()
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[0] == 1):
        plt.imshow(img_np, cmap='gray')
    elif img_np.ndim == 3 and img_np.shape[0] == 3:
        plt.imshow(img_np.transpose(1, 2, 0))
    else:
        plt.imshow(img_np)
    plt.axis('off')
    plt.title(f"Actual Class: {actual_class_name}")
    plt.show()

    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # --- Set model to evaluation mode ---
    model.eval()

    # --- Disable gradient computation for inference ---
    with torch.no_grad():
        # --- Forward pass ---
        outputs = model(image_tensor)

        # --- Convert logits to probabilities ---
        probabilities = torch.softmax(outputs, dim=1)

        # --- Print probabilities for each class ---
        for i, prob in enumerate(probabilities[0]):
            class_name = class_names[i] if i < len(class_names) else str(i)
            print(f"Class {class_name:10}: {prob:6.4f}")

        # --- Get predicted class and confidence ---
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_class.item()] if predicted_class.item() < len(dataset.dataset.dataset.classes) else str(predicted_class.item())
    print(f"Predicted Class: {predicted_class_name}, Confidence: {confidence.item():.4f}")
    
# --- Evaluation Utility ---
def evaluate_model(model, dataset):
    """
    Evaluates the model on the test dataset and prints accuracy.
    Returns the accuracy as a percentage.
    """
    # Collect all predictions and true labels from the test set
    true_labels = []
    predicted_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataset, desc="Evaluating Model"):
            images = images.unsqueeze(0)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend([labels])
            predicted_labels.extend(preds.cpu().numpy())

    # Define label names for MNIST
    label_names = dataset.dataset.dataset.classes
    n_classes = len(label_names)

    # Print classification report
    print(classification_report(true_labels, predicted_labels, target_names=label_names, labels=range(n_classes), zero_division=0))

    # Show confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(n_classes))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()