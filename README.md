<p align="center">
  <img src="image.png" alt="NMIMR MEDICT" width="200">
</p>

# NMIMR MEDICT
Medical Diagnosis using Computer Vision
## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [Lung Cancer Model](#lung-cancer-model)
  - [Kidney Cancer Model](#kidney-cancer-model)
  - [Brain Tumor Model](#brain-tumor-model)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Features

- Diagnose lung cancer, kidney cancer, and brain tumors from medical images
- Provide detailed information about each cancer type, including descriptions and precautions
- User-friendly web interface powered by Streamlit
- Utilizes pre-trained deep learning models for accurate predictions

## Installation

1. Clone the repository:
`git clone `
2. Install the required packages:
`pip install -r requirements.txt`
3. Download the pre-trained models from the following links and place them in the `models` directory:

- [Lung Cancer Model](https://example.com/lung_model.hdf5)
- [Kidney Cancer Model](https://example.com/kidney_model.hdf5)
- [Brain Tumor Model](https://example.com/brain_model.hdf5)

## Usage

1. Run the Streamlit app:
`python -m streamlit run App.py `
2. Upload a medical image (CT scan) through the web interface.
3. Select the desired cancer type from the sidebar.
4. The application will preprocess the image and provide a prediction along with the probability score.
5. If the prediction indicates the presence of cancer, the application will display relevant precautions and recommendations.

## Models

The deep learning models used in MeDiCT were trained on the following datasets:

- **Lung Cancer Model**: Trained on the [Lung Cancer Dataset from Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Kidney Cancer Model**: Trained on the [Kidney Cancer Dataset from Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- **Brain Tumor Model**: Trained on the [Brain Tumor Dataset from Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### Lung Cancer Model

The lung cancer model is based on the VGG16 architecture, a popular convolutional neural network model for image classification tasks. The VGG16 model was pre-trained on the ImageNet dataset and then fine-tuned on the lung cancer dataset.

## Models

### VGG16 Architecture

The VGG16 (Visual Geometry Group 16) is a popular convolutional neural network architecture proposed by researchers at the University of Oxford. It was introduced in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" and achieved excellent results on the ImageNet dataset.

The VGG16 architecture consists of 16 convolutional layers, followed by three fully connected layers. The convolutional layers are arranged in a series of blocks, with each block consisting of several convolutional layers with small 3x3 filters, followed by a max-pooling layer for spatial downsampling. The depth of the network increases as it progresses, starting with 64 filters in the first block and doubling with each subsequent block.

The VGG16 architecture is known for its simplicity and uniformity, as it uses the same filter size (3x3) and the same padding technique throughout the convolutional layers. This simplicity, combined with the depth of the network, allows the VGG16 to learn rich and hierarchical feature representations from input images.

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg" alt="VGG16 Architecture" width="500">
</p>

### Lung Cancer Model

The lung cancer model in MeDiCT is based on the VGG16 architecture. The pre-trained VGG16 model weights, obtained from training on the ImageNet dataset, are used as the starting point. The top classification layers of the VGG16 model are then replaced with new layers tailored for the lung cancer classification task.

The model architecture consists of the following layers:

1. **VGG16 Base Model**: The pre-trained VGG16 model without the top classification layers.
2. **Flatten Layer**: Flattens the output from the base model.
3. **Dropout Layer**: Applies dropout regularization to prevent overfitting.
4. **Dense Layer**: A fully connected layer with sigmoid activation for multi-label classification.

### Kidney Cancer Model

The kidney cancer model in MeDiCT also utilizes the VGG16 architecture as its backbone. Similar to the lung cancer model, the pre-trained VGG16 weights are used as the starting point, and the top classification layers are replaced with new layers specific to the kidney cancer classification task.

### Brain Tumor Model

The brain tumor model follows a similar approach, leveraging the VGG16 architecture and its pre-trained weights as the foundation. The top classification layers are replaced with new layers tailored for the brain tumor classification task.


## Data

The datasets used for training the models can be found in the following locations:

- [Lung Cancer Dataset](https://www.kaggle.com/mohamedhanyyy/chest-ctscan-images)
- [Kidney Cancer Dataset](https://www.kaggle.com/datasets/your-username/kidney-cancer-dataset)
- [Brain Tumor Dataset](https://www.kaggle.com/datasets/your-username/brain-tumor-dataset)

If you wish to retrain the models or experiment with different architectures, you can download the datasets from the provided links.

## Preprocessing

The image preprocessing steps include:

1. **Rescaling**: The pixel values of the input images are rescaled to the range [0, 1] by dividing by 255.
2. **Resizing**: The input images are resized to a fixed size of 350x350 pixels to ensure consistent input dimensions for the models.
3. **Data Augmentation**: For the lung cancer model, data augmentation techniques such as horizontal flipping, zooming, shearing, and shifting are applied to the training data to increase the diversity of the training set and improve model generalization.

## Training

The models were trained using the following techniques:

- **Transfer Learning**: The VGG16 model for lung cancer was pre-trained on the ImageNet dataset and then fine-tuned on the lung cancer dataset.
- **Early Stopping**: Early stopping was used to prevent overfitting by monitoring the validation accuracy and stopping the training when it no longer improved.
- **Model Checkpointing**: The best model weights were saved during training based on the highest validation accuracy.
- **Learning Rate Scheduling**: The learning rate was adjusted during training using a ReduceLROnPlateau callback to improve convergence.

## Contributing

Contributions are welcome! If you'd like to contribute to MeDiCT, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-feature-branch`
5. Submit a pull request

## Image Credits

The application uses the following images, which should be included in the repository.
