# Transfer Learning for Food Vision
***NOTE: For the detailed code please refer to the Collab version. Model scripts (data.setup.py, engine.py) are available in model_scripts.pynb section.***

What is **transfer learning?**
Transfer learning allows us to take the patterns (also called weights) another model has learned from another problem and use them for our own problem.

For example, we can take the patterns a computer vision model has learned from datasets such as ImageNet (millions of images of different objects) and use them to power our FoodVision Mini model.

Or we could take the patterns from a language model (a model that's been through large amounts of text to learn a representation of language) and use them as the basis of a model to classify different text samples.

The premise remains: find a well-performing existing model and apply it to your own problem.

![image](https://github.com/AICODER009/pytorch_transfer_learning/assets/133597851/9df026cd-7bad-4d7c-9f09-617e1900ea49)

# Why use transfer learning?
There are two main benefits to using transfer learning:

Can leverage an existing model (usually a neural network architecture) proven to work on problems similar to our own.

Can leverage a working model which has already learned patterns on similar data to our own. This often results in achieving great results with less custom data.

![image](https://github.com/AICODER009/pytorch_transfer_learning/assets/133597851/c00b5797-5943-4b52-89f1-ed623a344c47)

# Setting up a pretrained model(EfficientNet_B0)
The pretrained model we're going to be using is torchvision.models.efficientnet_b0().

The architecture is from the paper [ EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.](https://arxiv.org/abs/1905.11946)

``` weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights for ImageNet ```

our efficientnet_b0 comes in three main parts:

- features - A collection of convolutional layers and other various activation layers to learn a base representation of vision data (this base representation/collection of layers is often referred to as features or feature extractor, "the base layers of the model learn the different features of images").

- avgpool - Takes the average of the output of the features layer(s) and turns it into a feature vector.

- classifier - Turns the feature vector into a vector with the same dimensionality as the number of required output classes (since efficientnet_b0 is pretrained on ImageNet and because ImageNet has 1000 classes, out_features=1000 is the default).

![image](https://github.com/AICODER009/pytorch_transfer_learning/assets/133597851/5c17e07f-559a-452a-a5b4-e1b36b2ac011)
