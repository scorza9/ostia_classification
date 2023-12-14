### Neuroengineering project
# Ostia classification
Using deep learning on CCTA images, our aim is to solve a binary classification problem: with our Convolutional Neural Network, we made it able to determine whether a point inside or nearby a coronary artery belongs to a coronary ostium.

This project is part of the Neuroengineering course @ Politecnico di Milano

## Dataset
The dataset has been created by us, starting from CCTA 3D images. 
Using CAT08 and ASOCA datasets, we assigned ASOCA patients to training and validation, while CAT08 to testing. When performing splitting we ensured that images of the same patient would be found in the same set, given their correlation.  

## Preprocessing
To generate the network input, we initially aimed to divide patient volumes into smaller cubes, assigning labels based on proximity to coronary ostia. However, due to computational costs, we opted for a strategy where 50 points were randomly selected from the coronary artery graph, while another 50 were chosen from the ostia. Random translations were applied, and cubes were built around these points. For the training set, we added random rotations for augmentation. The result is a balanced dataset [of 24x24x24 patches], with isotropic voxel spacing of 0.5mm. Hounsfield units were clipped [between -800 and 1000] and standardized based on cube mean and standard deviation. 

## Network architecture
The subsequent step was to design the convolutional neural network. The reference architecture was taken from the literature. It consists in a fully convolutional neural network divided into a feature extraction network, composed of a stack of 4 dilated convolutions and a feature classification and regressor network, composed of convolutions of stride 1. 

Our models had the goal to minimize inference time while retaining a high performance. Each implementation therefore maintained the stack of 4 dilated convolutions, which allows to grow the receptive field exponentially, while the number of parameters increases linearly. 

# FASTCONV3DNET_V1:
The first model developed was therefore a simple adaptation of the previous one with a reduced number of filters for each layer and a dense network as the classification head. 

# FASTCONV3DNET_V2:  

The evolution of the model was a smaller implementation featuring the channel attention mechanism in between each convolutional layer. This allows to weight each feature map based on the result of a squeeze and excitation block. 

Results from these models show a slightly better performance and faster inference by the second model. 

# DEPTHWISE-SEPARABLE CONVOLUTION: 

To speed up inference, a custom Depthwise-Separable 3D Convolution was implemented. It enhances computational efficiency by splitting the convolution into two steps: a depth-wise convolution processing each channel independently and a point-wise convolution combining information across channels [using 1x1x1 convolutions.]  