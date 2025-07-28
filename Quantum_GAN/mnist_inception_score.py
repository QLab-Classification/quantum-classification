import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

# Fix the MNIST Classifier to handle variable input sizes
class MNISTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Ensure input is grayscale and properly shaped
        if x.shape[1] == 3:  # If RGB, convert to grayscale
            x = x.mean(dim=1, keepdim=True)
        elif len(x.shape) == 3:  # If (N, H, W), add channel dimension
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        
        # Use adaptive pooling to get consistent size
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MNISTInceptionScore:
    """
    Calculate the Inception Score (IS) for MNIST using a convolutional classifier.
    
    The Inception Score measures both quality and diversity of generated images:
    IS(G) = exp(E_{x̃~G} [KL(p(y|x̃)||p(y))])
    
    Higher scores indicate better quality and diversity.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.classifier = self._load_mnist_classifier()
        self.transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # MNIST normalization
        ])
        
    def _load_mnist_classifier(self):
        """Load or create MNIST classifier with ~99.3% accuracy."""
        classifier = MNISTClassifier(num_classes=10)
        
        # Try to load pre-trained weights if available
        try:
            classifier.load_state_dict(torch.load('mnist_classifier_99_3.pth', map_location=self.device))
            print("Loaded pre-trained MNIST classifier (~99.3% accuracy)")
        except FileNotFoundError:
            print("No pre-trained classifier found. Using untrained model.")
            print("Run 'train_mnist_classifier.py' first to train the classifier.")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            print("Using untrained model.")
        
        classifier.eval()
        return classifier.to(self.device)
    
    def _preprocess_images(self, images):
        """
        Preprocess images for MNIST classifier.
        
        Args:
            images: Tensor of shape (N, C, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            Preprocessed images ready for MNIST classifier
        """
        # Convert to [0, 1] range if in [-1, 1]
        if images.min() < 0:
            images = (images + 1) / 2
            
        # Ensure grayscale (single channel)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        elif images.shape[1] == 1:
            pass  # Already grayscale
        else:
            # If no channel dimension, add it
            images = images.unsqueeze(1)
            
        # Resize to a reasonable size for the classifier (e.g., 64x64)
        if images.shape[2] != 64 or images.shape[3] != 64:
            images = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
            
        return images.to(self.device)
    
    def _get_classifier_predictions(self, images, batch_size=32):
        """
        Get MNIST classifier predictions for images.
        
        Args:
            images: Tensor of preprocessed images
            batch_size: Batch size for processing
            
        Returns:
            Softmax probabilities from MNIST classifier
        """
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                outputs = self.classifier(batch)
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
                
        return np.concatenate(predictions, axis=0)
    
    def calculate_inception_score(self, generator, num_samples=5000, batch_size=32, splits=10):
        """
        Calculate the Inception Score for a trained generator.
        
        Args:
            generator: Trained generator model
            num_samples: Number of samples to generate for evaluation
            batch_size: Batch size for processing
            splits: Number of splits for calculating mean and std
            
        Returns:
            tuple: (mean_inception_score, std_inception_score)
        """
        generator.eval()
        
        # Generate samples
        print(f"Generating {num_samples} samples for Inception Score calculation...")
        all_predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - i)
                z = torch.randn(current_batch_size, generator.layer1[0].in_features).to(self.device)
                fake_images = generator(z)
                
                # Preprocess images
                processed_images = self._preprocess_images(fake_images)
                
                # Get classifier predictions
                predictions = self._get_classifier_predictions(processed_images, batch_size)
                all_predictions.append(predictions)
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate Inception Score
        print("Calculating Inception Score...")
        scores = []
        
        for i in range(splits):
            part = all_predictions[i * (num_samples // splits):(i + 1) * (num_samples // splits)]
            kl_divergence = entropy(part.T, base=2)
            scores.append(np.exp(np.mean(kl_divergence)))
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score
    
    def calculate_inception_score_from_images(self, images, splits=10):
        """
        Calculate Inception Score from a set of images.
        
        Args:
            images: Tensor of images in format (N, C, H, W)
            splits: Number of splits for calculating mean and std
            
        Returns:
            tuple: (mean_inception_score, std_inception_score)
        """
        print("Preprocessing images for Inception Score calculation...")
        processed_images = self._preprocess_images(images)
        
        print("Getting classifier predictions...")
        predictions = self._get_classifier_predictions(processed_images)
        
        # Calculate Inception Score
        print("Calculating Inception Score...")
        scores = []
        num_samples = len(predictions)
        
        for i in range(splits):
            part = predictions[i * (num_samples // splits):(i + 1) * (num_samples // splits)]
            kl_divergence = entropy(part.T, base=2)
            scores.append(np.exp(np.mean(kl_divergence)))
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score

def evaluate_mnist_gan_performance(generator, num_samples=5000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate MNIST GAN performance using Inception Score.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate for evaluation
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("Starting MNIST GAN performance evaluation...")
    
    # Initialize MNIST Inception Score calculator
    is_calculator = MNISTInceptionScore(device=device)
    
    # Calculate Inception Score
    mean_is, std_is = is_calculator.calculate_inception_score(
        generator, num_samples=num_samples
    )
    
    results = {
        'inception_score_mean': mean_is,
        'inception_score_std': std_is,
        'inception_score': f"{mean_is:.3f} ± {std_is:.3f}",
        'num_samples': num_samples
    }
    
    print(f"Inception Score: {mean_is:.3f} ± {std_is:.3f}")
    print(f"Number of samples used: {num_samples}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("MNIST Inception Score Evaluation for GANs")
    print("=" * 50)
    print("This implementation uses a convolutional classifier")
    print("instead of Inception-v3 for grayscale MNIST data.")
    print("=" * 50)
    
    # This would be used after training your GAN
    # from your_gan_file import Generator, generator
    
    # Example evaluation (uncomment when you have a trained generator)
    # results = evaluate_mnist_gan_performance(generator, num_samples=5000)
    # print(f"Final Inception Score: {results['inception_score']}") 