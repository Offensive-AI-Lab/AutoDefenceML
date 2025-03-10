# **Writign an AutoDefenceML Plugin**
The AutoDefenceML platform supports easy plug and play additions of new attacks and defences for the Model Evaluation (adversarial examples). 
All new attacks and defences must be compatible with the art toolbox.

Here we demonstrate how to make these plugins:

## **Overview**
This tutorial will guide you through:
1. Implementing a **toy attack** that adds random noise to images.
2. Implementing a **preprocessor defense** that removes noise from images.

We'll use the **Fast Gradient Sign Method (FGSM)** implementation in ART as a reference.

---

## **1. Implementing a Custom Evasion Attack**
To create a new attack, we must:
- Inherit from `EvasionAttack`
- Implement the `generate()` method

### **Step 1: Inherit from `EvasionAttack`**
Create a new attack class called `NoiseAttack` that adds random noise.

```python
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
import numpy as np

class NoiseAttack(EvasionAttack):
    """
    A simple evasion attack that adds random noise to images.
    """

    attack_params = EvasionAttack.attack_params + ["noise_level"]
    _estimator_requirements = (BaseEstimator,)

    def __init__(self, estimator, noise_level=0.1):
        """
        :param estimator: A trained classifier.
        :param noise_level: The magnitude of noise to be added (default 0.1).
        """
        super().__init__(estimator)
        self.noise_level = noise_level
        NoiseAttack._check_params(self)

    def generate(self, x, y=None):
        """
        Generate adversarial examples by adding random noise.
        
        :param x: Input images (numpy array).
        :param y: (Optional) Target labels.
        :return: Adversarially perturbed images.
        """
        noise = np.random.uniform(-self.noise_level, self.noise_level, x.shape)
        x_adv = np.clip(x + noise, 0, 1)  # Ensure values remain valid
        return x_adv

    def _check_params(self):
        """
        Validate parameters.
        """
        if not (0 <= self.noise_level <= 1):
            raise ValueError("noise_level must be between 0 and 1.")
```

---

## **2. Implementing a Preprocessor Defense**
To create a new defense, we must:
- Inherit from `Preprocessor`
- Implement the `__call__()` method

### **Step 2: Inherit from `Preprocessor`**
Create a new preprocessor class that **smooths images** to remove adversarial noise.

```python
from art.defences.preprocessor import Preprocessor
import scipy.ndimage

class DenoisingPreprocessor(Preprocessor):
    """
    A simple preprocessor that applies Gaussian blurring to denoise adversarial images.
    """

    def __init__(self, sigma=0.5, apply_fit=False, apply_predict=True):
        """
        :param sigma: The standard deviation for Gaussian blur.
        :param apply_fit: Whether to apply the defense during model training.
        :param apply_predict: Whether to apply the defense during inference.
        """
        super().__init__(apply_fit=apply_fit, apply_predict=apply_predict)
        self.sigma = sigma

    def __call__(self, x, y=None):
        """
        Apply Gaussian blurring to remove adversarial noise.

        :param x: Input images.
        :param y: Labels (not modified).
        :return: Denoised images.
        """
        x_denoised = np.array([scipy.ndimage.gaussian_filter(img, sigma=self.sigma) for img in x])
        return x_denoised, y

    def estimate_gradient(self, x, grad):
        """
        Provide a dummy gradient estimation.
        """
        return grad
```

---

## **3. Testing the Attack and Defense**
Now, let's test our attack and defense with a simple example.

### **Step 3: Load a Model and Apply the Attack**
```python
from art.estimators.classification import TensorFlowV2Classifier
import tensorflow as tf
import numpy as np

# Load a simple model (e.g., trained MNIST classifier)
model = tf.keras.models.load_model("path_to_your_model.h5")

classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=10,
    input_shape=(28, 28, 1),
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
)

# Load sample images (e.g., MNIST)
x_test = np.random.rand(10, 28, 28, 1)  # Fake data for example
y_test = np.eye(10)[np.random.choice(10, 10)]  # Fake labels

# Apply the attack
attack = NoiseAttack(classifier, noise_level=0.1)
x_adv = attack.generate(x_test)
```

---

### **Step 4: Apply the Preprocessor Defense**
```python
# Apply the preprocessor to remove noise
denoiser = DenoisingPreprocessor(sigma=1.0)
x_denoised, _ = denoiser(x_adv)

# Evaluate the classifier on original, adversarial, and denoised inputs
acc_orig = np.mean(np.argmax(classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1))
acc_adv = np.mean(np.argmax(classifier.predict(x_adv), axis=1) == np.argmax(y_test, axis=1))
acc_denoised = np.mean(np.argmax(classifier.predict(x_denoised), axis=1) == np.argmax(y_test, axis=1))

print(f"Accuracy on original data: {acc_orig:.2f}")
print(f"Accuracy on adversarial data: {acc_adv:.2f}")
print(f"Accuracy after denoising: {acc_denoised:.2f}")
```

---

## **Conclusion**
In this tutorial, we:
- Implemented a **toy attack** that adds random noise.
- Implemented a **preprocessor defense** that smooths images to remove noise.
- Tested both attack and defense using a classifier.

## **TODO**
- steps for adding to framework
