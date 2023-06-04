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

# Freezing the base model and shifting the output layer to suit the given problem
The process of transfer learning usually consists of two steps: 

1)freeze some base layers of a pretrained model (mainly the features section;

2) add and change the output layers (named head/classifier layers respectively) to suit your problem.

![image](https://github.com/AICODER009/pytorch_transfer_learning/assets/133597851/f042cb77-e78f-47cb-a556-08bf7ff127df)

The actual *torchvision.models.efficientnet_b0()* comes with ```out_features=1000``` because there are *1000 classes* in ImageNet, the dataset it was taught on. Nevertheless, for our needs, for the food vision problem, we only need ```out_features=3```.

**Note:** To freeze layers indicates to hold them how they are during training. For illustration, if your model has pre trained layers, to freeze them would be to declare, "Don't alter any of the marks in these layers during training, keep them how they are." In nature, we'd like to save the pre-trained patterns/weights our model has memorized from ImageNet as a backbone and then only adjust the output layers.

We can freeze all of the parameters/layers in the features section by specifying the attribute ```requires_grad=False```.

``` ruby
# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False
```
Now we've got a pretraiend model that's semi-frozen and has a customized classifier and now we can train it.

**Note:** The more trainable parameters a model has, the more compute longer/power it takes to train. Freezing the base layers of our model and exiting it with less trainable parameters signifies our model should train quite fast. This is one huge advantage of transfer learning, taking the already memorized parameters/patterns of a model trained on a problem comparable to yours and only squeezing the outputs barely to fit your problem.
