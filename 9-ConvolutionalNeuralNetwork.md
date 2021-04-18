# CNN (Convolutional Neural Network)

Convolutional Neural Netowrk have a different architecture than regular Neural Networks. Regular Neural Network transform an input by putting it through a series of hidden layers. Every layer is made up of a set of neurons, where each layer is fully connected to all neurons in the layer before. Finally, there is a last fully connected layer - the output layer - that represent the predictions.

Convolutional Neural Network are a bit different. First of all, the layers are organised in 3 dimensions: width, height and depth. Further, the neurons in one layer do not connect to all the neurons
in the next layer but only to a small region of it. Lastly, the final output will be reduced to a single vector of probability scores, organized along the depth dimension.

![Dgy6hBvOvAWofkrDM8BclOU3E3C2hqb25qBb](https://user-images.githubusercontent.com/23405520/115138480-0c573d00-a04a-11eb-8978-b8d741f26a48.png)

        ` Normal NN vs CNN`
        
CNN have two components:

- <b>  The Hidden layers / Feature extraction part. </b>

In this part, the network will perform a series of convolutions and pooling operations during which the features are detected. If you had a picture of a zebra, this is the part where the network would recognize its stripes, two ears and four legs.


- <b> The Classification Part </b>

Here, the fully connected layers will serve as a classifier on top of these extracted features. They will assign a probability for the object on the image being what the algorithm predicts it is.


![dobVrh3SGyqQraM2ogi-P3VK2K-LFsBm7RLO](https://user-images.githubusercontent.com/23405520/115138549-85569480-a04a-11eb-95f9-9ceb7bc6a10b.png)


### Feature Extraction
Convolution is one of the main building blocks of a CNN. The term convolution refers to the mathematical combination of two functions to produce a third function. It merges two sets of information.

In the case of a CNN, the convolution is performed on the input data with the use of a filter or kernel (these terms are used interchangeably) to then produce a feature map.

We execute a convolution by sliding the filter over the input. At every location, a matrix multiplication is performed and sums the result onto the feature map.

In the animation below, you can see the convolution operation. You can see the filter (the green square) is sliding over our input (the blue square) and the sum of the convolution goes into the feature map (the red square).

The area of our filter is also called the receptive field, named after the neuron cells! The size of the filter is 3 * 3.

![Htskzls1pGp98-X2mHmVy9tCj0cYXkiCrQ4t](https://user-images.githubusercontent.com/23405520/115138663-25acb900-a04b-11eb-9cfb-b9fab5361c19.gif)

            Left: the filter slides over the input
            Right: the result is summed and added to the feature map.
            

This is just a operation in 2D, but in reality convolutions are performed in 3D. Each image is namely represented as a 3D matrix with a dimension for width, height and depth. Depth is a dimension because of the colours channel used in an image (RGB).

![Gjxh-aApWTzIRI1UNmGnNLrk8OKsQaf2tlDu](https://user-images.githubusercontent.com/23405520/115138723-76bcad00-a04b-11eb-9d17-eb4f979d7493.png)


                The filter slides over the input and performs its output on the new layer. 
                
                
We perform numerous convolutions on our input, where each operation uses a different filter This results in different feature maps. In the end, we take all of these feature maps and put them together as the final output of the convolution layer.

Just like any other Neural Network, we use an <b> activation function </b> to make our output non-linear. In the case of a Convolutional Neural Network, the output of the convolution will be passed through the activation function. This could be the <b> ReLU </b> activation function.

#### Stride

Stride is the size of the step the convolution filter moves each time. A stride size is usually 1, meaning the filter slides pixel by pixel. By increasing the stride size, your filter is sliding over the input with a larger interval and thus has less overlap between the cells.

The animation below shows stride size 1 in action.

![d0ufdQE7LHA43cdSrVefw2I9DFceYMixqoZJ](https://user-images.githubusercontent.com/23405520/115138823-15e1a480-a04c-11eb-934e-9863a2044b7c.gif)


Because the size of the feature map is always smaller than the input, we have to do something to prevent our feature map from shrinking. This is where we use <b> padding </b>

A layers of zero-value pixels is added to surround the input with zeros, so that our feature map will not shrink. In addition to keeping the spatial size constant after performing convolution, padding also improves performance and makes sure the kernel and stride size will fit the input.

#### Pooling

After a convolution layer, it is common to add a pooling layer in between CNN layers. The function of pooling to continously reduce the dimensionality to reduce the number of parameters and computation in the network . This shortness the training time and controls overfitting.

The most frequent type of pooling is <b> max pooling </b>, which takes the maximum value in each window. These window size need to be specified beforehand. This decrease the feature map size while at the same time keeping the significant information.

![96HH3r99NwOK818EB9ZdEbVY3zOBOYJE-I8Q](https://user-images.githubusercontent.com/23405520/115138934-c64fa880-a04c-11eb-9768-be9d3bdc79ad.png)

              Max pooling takes the largest values.
              
              
Thus when using a CNN, the four important <b> hyperparameters </b> we have to decide on are:

- the kernel size.
- the filter count (that is, how many filters do we want to use).
- stride (how big are the steps of the filter)
- padding


A nice way to visualize a convolutional layer is shown below. 

![gb08-2i83P5wPzs3SL-vosNb6Iur5kb5ZH43](https://user-images.githubusercontent.com/23405520/115138990-14fd4280-a04d-11eb-8c83-9bb84d738909.gif)

        How convolution works with K = 2 filters, each with a spatial extent F = 3, strides, S = 2 and input padding P = 1.
        
        
### Classification

After the convolution and pooling layers, our classification part consists of a few fully connected layers. However, these fully connected layers can only accept 1 Dimensional data. To convert our 3D data to 1D, we use the function <b> flatten </b> in Python. This essentially arranges our 3D volume into a 1D Vector.

The last layers of a Convolution NN are fully connected layers. Neurons in a fully connected layer have full connections to all activations in the previous layer. This part is in principle the same as a regular Neural Network.
