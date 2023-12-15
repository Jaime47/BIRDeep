<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F493790%2Fc8dbb8ea2de8b1aaa5dee7a3e05228ba%2Fbird.jpg?generation=1700057120700150&alt=media" lt="AML project banner" title="Applied Machine Learning - Feather in Focus" style="object-fit: cover; display: block; margin-left: auto; margin-right: auto; width: 50%" />

# Applied Machine Learning - Project: Feather in Focus! (BIRDeep)

This is the GitHub repository for the Applied Machine Learning Project: Feather in Focus!
This project is associated with a private Kaggle competition, and you can find more details about the competition [here](https://www.kaggle.com/competitions/feather-in-focus/overview).

## Description
The goal of the challenge is to classify the bird image and find the name of the bird! This dataset contains 200 bird species, and the goal is to achieve a high accuracy in predicting the birds!

## Introduction
BirdDeep's model strives for increased robustness by decoupling forward and background image subsets. This is achieved through background removal, comple-mented by other accuracy enhancing features including:
- Data Balancing
- Data augmentation
- Hyperparameter Tuning
- Use of pre-trained Models

## Research
Noise or signal: The role of image backgrounds in object recognition presents several background removal methods. BIRDeep uses the Only-FG format in Figure 1. This choice eliminates background interactions, enabling the model to concentrate solely on bird characteristics.

<img src="https://i.imgur.com/dm0zPOk.png">

*Figure 1: Obtained from Xiao, K., Engstrom, L., Ilyas, A., & Madry, A. (2020). Noise or signal: The role of image backgrounds in object recognition. arXiv preprint arXiv:2006.09994.*

The featured pre-trained model is Xception. This model employs depthwise separable convolution: a pointwise convolution succeeded by a depthwise convolution. 

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*VvBTMkVRus6bWOqrK1SlLQ.png">

*Figure 2: Xception convolution procedure. A 1x1 pointwise convolution is performed to then apply nxn depthwise convolutions.*

## Data Augmentation
To emulate the Only-FG image format, the system performs interactive foreground selection using GrabCut. This method introduces a simplicity tradeoff, occasionally yielding suboptimal results due to a predefined cutting area. In BIRDeep, a rectangular boundary (50, 50, 200, 200) is set over 299x299 pixel images, as illustrated in Figure 3. 

<img src="https://i.imgur.com/YrmOGA0.png">

*Figure 3: BirdDeep’s foreground and background extraction process based on GrabCut masking.*

Each foreground only image is incorporated to the training dataset alongside four copies subjected to data augmentation.
- Image horizontal flip
- Image 90 degrees rotation
- Image contrast increased by 50%
- Image contrast decreased by 50%

<img src="https://i.imgur.com/MYnPypb.png">

*Figure 4: BirdDeep’s data augmentation processes applied over a single image.*

## Results

### Baseline Model
The [Baseline Model](Baseline20%Model/Baseline_Model.ipynb) achieved a test accuracy of 58%.

### Mixed Model
The Mixed Model approach can be found [here](Mixed20%Model) and involves three jupyter notebooks:
- [Generating Subfolder](Mixed20%Model/generating_subfolder.ipynb) - Generates the needed folder structure for the Mixed Model
- [Data Balancing](Mixed20%Model/data_balancing.ipynb) - Performs data balancing to only use 27 files
- [Mixed Model](Mixed20%Model/model.ipynb) achieved a test accuracy of 60%. This approach includes mainly augmented Only-FG images, but also some raw images.

All the other Branches were created for testing purposes and can be ingnored regarding the sumbission for Applied Machine Learning.

## Contact
| Name                  | Email                                |
|-----------------------|--------------------------------------|
| Alina Baciu           | alina.baciu@student.uva.nl           |
| Thomas Erhard         | thomas.erhard@student.uva.nl         |
| Jaime Pons            | jaime.pons@student.uva.nl            |
| Leonardo Provenzano   | leonardo.provenzano@student.uva.nl   |

Project Link: https://github.com/Jaime47/BIRDeep

### Images Loading
To facilitate the efficient loading of images, we developed a function that creates distinct subfolders for each class and organizes the images accordingly. To streamline the process of transferring the dataset across different machines, such as Colab and Snellius, we uploaded the directory containing both train and test images to [Roboflow](https://roboflow.com/). Leveraging Roboflow's export capabilities, we were able to easily move the images to our desired destinations whenever needed.
