# Example usage of Inception Score evaluation for your trained GAN
import torch
from inception_score import evaluate_gan_performance

def evaluate_trained_gan(generator_path, latent_dim=100, num_samples=5000):
    """
    Evaluate a trained GAN using Inception Score.
    
    Args:
        generator_path: Path to saved generator model
        latent_dim: Latent dimension used in training
        num_samples: Number of samples to generate for evaluation
    """
    
    # Load the trained generator
    print(f"Loading generator from: {generator_path}")
    
    # Load the saved model
    checkpoint = torch.load(generator_path, map_location='cpu')
    
    # Initialize generator (adjust the import based on your Generator class)
    from ClassicalGAN import Generator  # Adjust this import to match your file
    generator = Generator(latent_dim)
    
    # Load the state dict
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Evaluate using Inception Score
    print("Evaluating GAN performance...")
    results = evaluate_gan_performance(generator, num_samples=num_samples)
    
    print("\n" + "="*50)
    print("GAN EVALUATION RESULTS")
    print("="*50)
    print(f"Inception Score: {results['inception_score']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Mean IS: {results['inception_score_mean']:.3f}")
    print(f"Std IS: {results['inception_score_std']:.3f}")
    print("="*50)
    
    return results

# Example usage:
if __name__ == "__main__":
    # Replace with your actual model path
    model_path = "models/gan_final_at_time_1234567890.12.pth"
    
    try:
        results = evaluate_trained_gan(model_path, latent_dim=100, num_samples=5000)
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure to:")
        print("1. Update the model_path to your actual saved model")
        print("2. Adjust the latent_dim to match your training")
        print("3. Import your Generator class correctly") 