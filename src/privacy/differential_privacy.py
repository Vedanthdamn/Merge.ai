"""
Differential Privacy for Healthcare Machine Learning

Differential Privacy (DP) adds carefully calibrated noise to prevent
re-identification of individual patients in the training data.

Key Concept:
Adding or removing a single patient's data should not significantly
change the model's output or behavior.

How DP Works:
1. During training, noise is added to gradients or model updates
2. Noise magnitude is controlled by privacy budget (epsilon)
3. Lower epsilon = more privacy but potentially lower accuracy
4. Higher epsilon = less privacy but better accuracy

Privacy Guarantee:
An adversary cannot determine if a specific patient was in the training
dataset, protecting individual privacy while enabling model learning.

Example:
Without DP: Model might memorize rare patient cases
With DP: Noise obscures individual contributions, preventing memorization
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for healthcare ML.
    
    This class provides methods to add noise to gradients and model
    parameters, ensuring individual patient data cannot be extracted
    from the trained model.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 clip_norm: float = 1.0):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon (float): Privacy budget. Lower = more private.
                - epsilon < 1: Strong privacy
                - epsilon 1-10: Moderate privacy
                - epsilon > 10: Weak privacy
            delta (float): Probability of privacy breach. Should be very small.
            clip_norm (float): Gradient clipping threshold.
                Prevents any single patient from having too much influence.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Calculate noise scale based on privacy parameters
        self.noise_scale = self._calculate_noise_scale()
        
        print(f"[Differential Privacy] Initialized")
        print(f"  Epsilon (privacy budget): {epsilon}")
        print(f"  Delta (failure probability): {delta}")
        print(f"  Clip norm: {clip_norm}")
        print(f"  Noise scale: {self.noise_scale:.4f}")
    
    def _calculate_noise_scale(self) -> float:
        """
        Calculate noise scale for Gaussian mechanism.
        
        Gaussian mechanism:
        - Adds Gaussian noise to gradients
        - Noise calibrated to sensitivity and privacy budget
        
        Returns:
            Noise standard deviation
        """
        # Simplified calculation: sensitivity / epsilon
        # In practice, would use more sophisticated methods
        sensitivity = self.clip_norm * 2  # L2 sensitivity
        noise_scale = sensitivity / self.epsilon
        return noise_scale
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clip gradients to bound individual patient influence.
        
        Why clip?
        - Limits how much any single patient can affect the model
        - Prevents outlier patients from dominating learning
        - Essential for differential privacy guarantees
        
        Process:
        1. Compute L2 norm of all gradients
        2. If norm > threshold, scale down gradients
        3. Ensures no single example has excessive influence
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Clipped gradients
        """
        # Compute global L2 norm across all gradients
        global_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
        
        # Clip if necessary
        if global_norm > self.clip_norm:
            clip_factor = self.clip_norm / global_norm
            clipped_gradients = [g * clip_factor for g in gradients]
            return clipped_gradients
        
        return gradients
    
    def add_noise_to_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise to gradients for differential privacy.
        
        Why add noise?
        - Makes it impossible to determine if a specific patient
          was in the training data
        - Prevents model from memorizing individual cases
        - Provides formal privacy guarantees
        
        Process:
        1. Clip gradients to bound sensitivity
        2. Add Gaussian noise scaled to privacy parameters
        3. Noisy gradients still point in approximately correct direction
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Noisy gradients with privacy guarantees
        """
        # Step 1: Clip gradients
        clipped_gradients = self.clip_gradients(gradients)
        
        # Step 2: Add Gaussian noise
        noisy_gradients = []
        for grad in clipped_gradients:
            noise = np.random.normal(0, self.noise_scale, grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def add_noise_to_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add noise to model parameters (alternative DP approach).
        
        Instead of noisy gradients, directly perturb model weights.
        Useful for federated learning where model updates are shared.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Noisy parameters
        """
        noisy_parameters = []
        for param in parameters:
            noise = np.random.normal(0, self.noise_scale * 0.1, param.shape)
            noisy_param = param + noise
            noisy_parameters.append(noisy_param)
        
        return noisy_parameters
    
    def compute_privacy_spent(self, num_steps: int, batch_size: int, 
                             dataset_size: int) -> Tuple[float, float]:
        """
        Compute privacy budget spent during training.
        
        Privacy accounting:
        - Each training step consumes privacy budget
        - More steps = more privacy loss
        - Smaller batches = more privacy loss per epoch
        
        Args:
            num_steps (int): Number of training steps
            batch_size (int): Batch size
            dataset_size (int): Total dataset size
            
        Returns:
            Tuple of (epsilon_spent, delta)
        """
        # Simplified privacy accounting
        # In practice, would use tools like TensorFlow Privacy
        sampling_probability = batch_size / dataset_size
        epsilon_per_step = self.epsilon * sampling_probability
        epsilon_spent = epsilon_per_step * num_steps
        
        return epsilon_spent, self.delta
    
    def get_privacy_report(self, num_epochs: int, batch_size: int,
                          dataset_size: int) -> str:
        """
        Generate human-readable privacy report.
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            dataset_size (int): Dataset size
            
        Returns:
            Privacy report string
        """
        steps_per_epoch = dataset_size // batch_size
        total_steps = num_epochs * steps_per_epoch
        epsilon_spent, delta = self.compute_privacy_spent(
            total_steps, batch_size, dataset_size
        )
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║           DIFFERENTIAL PRIVACY REPORT                     ║
╚═══════════════════════════════════════════════════════════╝

Privacy Parameters:
  - Initial epsilon (ε): {self.epsilon}
  - Delta (δ): {self.delta}
  - Gradient clip norm: {self.clip_norm}

Training Configuration:
  - Number of epochs: {num_epochs}
  - Batch size: {batch_size}
  - Dataset size: {dataset_size}
  - Total training steps: {total_steps}

Privacy Analysis:
  - Estimated epsilon spent: {epsilon_spent:.2f}
  - Privacy guarantee: ({epsilon_spent:.2f}, {delta})-DP

Interpretation:
  - Lower epsilon = Stronger privacy protection
  - This model provides {"strong" if epsilon_spent < 1 else "moderate" if epsilon_spent < 10 else "weak"} privacy
  - Individual patient data is protected from re-identification

Note: This is a simplified privacy accounting. Production systems
should use formal privacy accounting tools like TensorFlow Privacy
or Opacus for rigorous guarantees.
"""
        return report


class DPOptimizer:
    """
    Differential privacy optimizer wrapper.
    
    Wraps standard optimizer to add DP during training.
    """
    
    def __init__(self, base_optimizer, dp_mechanism: DifferentialPrivacy):
        """
        Initialize DP optimizer.
        
        Args:
            base_optimizer: Underlying optimizer (e.g., Adam)
            dp_mechanism: Differential privacy mechanism
        """
        self.base_optimizer = base_optimizer
        self.dp = dp_mechanism
    
    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients with differential privacy.
        
        Process:
        1. Extract gradients
        2. Clip gradients per-example
        3. Add calibrated noise
        4. Apply noisy gradients to model
        
        Args:
            grads_and_vars: List of (gradient, variable) tuples
        """
        # Extract gradients
        gradients = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]
        
        # Add differential privacy
        noisy_gradients = self.dp.add_noise_to_gradients(gradients)
        
        # Apply noisy gradients
        self.base_optimizer.apply_gradients(zip(noisy_gradients, variables))


def demonstrate_dp():
    """
    Demonstrate differential privacy on mock gradients.
    """
    print("="*60)
    print("Differential Privacy Demonstration")
    print("="*60)
    
    # Create mock gradients
    gradients = [
        np.random.randn(10, 5),
        np.random.randn(5, 1)
    ]
    
    print(f"\nOriginal gradient norms:")
    for i, g in enumerate(gradients):
        print(f"  Layer {i}: {np.linalg.norm(g):.4f}")
    
    # Apply DP with different privacy levels
    privacy_levels = [
        (0.1, "Very Strong Privacy"),
        (1.0, "Strong Privacy"),
        (10.0, "Moderate Privacy")
    ]
    
    for epsilon, description in privacy_levels:
        print(f"\n{'-'*60}")
        print(f"{description} (epsilon = {epsilon})")
        print(f"{'-'*60}")
        
        dp = DifferentialPrivacy(epsilon=epsilon, clip_norm=1.0)
        noisy_gradients = dp.add_noise_to_gradients(gradients)
        
        print("Noisy gradient norms:")
        for i, g in enumerate(noisy_gradients):
            print(f"  Layer {i}: {np.linalg.norm(g):.4f}")
        
        # Show privacy report
        print(dp.get_privacy_report(
            num_epochs=10,
            batch_size=32,
            dataset_size=1000
        ))


def main():
    """Main demonstration function."""
    demonstrate_dp()


if __name__ == "__main__":
    main()
