# FashionMNIST
Image Classification on FashionMNIST Dataset with Data Analysis


## Introduction
This project aims to develop an AI solution for NITEX to identify and classify sustainable apparel products using the Fashion MNIST dataset. The solution includes data analysis, model development, and a human-in-the-loop approach to enhance accuracy. The project follows the guidelines provided and focuses on aligning with NITEX's vision for sustainable fashion.

## Project Structure

### Analysis
  Analysis.ipynb: Jupyter Notebook for exploring the dataset and documenting insights.
  Findings: 
  1. The raw csv contains the labels and the pixel intensity, The first column holds the label and next 784 column holds all the pixel value of the image.
  2. The shape of the feature is 28x28x1, height and width of the images are 28x28 and has a channel 1 which indicates these images are grayscale
  3. In the class distribution we can see, there are 10 labels (0->9) and each class contains 6,000 images making a total of 6000x10= 60,000 images.
  4. The histogram on mean pixel intensity is left skewed (or negatively skewed) which indicates that the majority of the images in the dataset have lower mean pixel values, or they are relatively darker.
  5. In the context of grayscale images, lower pixel values represent darker shades of gray, whereas higher pixel values represent lighter shades.
  6. The histogram on standard deviation suggests that the pixel values within each image are relatively consistent and do not vary widely from the mean intensity.
  7. I put a visualization for first 10 images of each label. According to the dataset descrioption the labels <ul>  <li>0 T-shirt/top </li> <li>1 Trouser</li> <li>2 Pullover </li> <li>3 Dress </li> <li>4 Coat </li><li>5 Sandal </li> <li>6 Shirt </li> <li>7 Sneaker </li> <li>8 Bag </li> <li>9 Ankle boot </li> </ul> Juidging by first 10 images, we can already tell the main mis-classification will occure between label 0 (T-shirt/top) and label 6 (6 Shirt), as they are both very similar, with very few details which could alsobe oversighted by humaneyes.



### Model Architecture & Training
  Training.ipynb: Jupyter Notebook for designing and training the machine learning model.
  Motivation: 
  <ul>I started with building the model with two conv2d layers with filters=8. then added and removed BatchNormlization, Maxpool2d layers. First I kept the conv2d layer constant and changed the values and the frequencies for Maxpool2d and dropout layes, in order to find the best fit. After I was happy with the result where my model is in the bestfit (consistency between validation and training loss). I started to aim for the higher accuracies by increasing the conv2 layers. While I increased more conv2 layers, I had to change some values in the dropout layer to minimize the gap between training and validation loss. I kept a note on which layer is contributing what to my architecture, which helped me find better results with the model.<ul>



evaluate_model.py: Script to evaluate the trained model on a given dataset folder.
model_summary.txt: Text file containing the summary of the trained model's architecture.
output.txt: Text file generated after model evaluation, including model summary, evaluation metrics, and insights.
Data Analysis
In the data_analysis.ipynb notebook, we explored the Fashion MNIST dataset to understand the distribution of classes and the nature of images. Key insights and patterns observed during the analysis are documented in the notebook.

Model Development
The model_training.ipynb notebook contains the code for designing and training a machine learning model using the Fashion MNIST dataset. Various architectures and approaches were explored to achieve accurate classification of sustainable apparel products.

Human-in-the-Loop Approach
To enhance human-in-the-loop efficiency, we propose a method where human expertise can be combined with the model's predictions. Users can review uncertain predictions flagged by the model and provide corrections, improving the overall accuracy of the system.

Model Evaluation
To evaluate the trained model, run the evaluate_model.py script with the following command:

bash
Copy code
python evaluate_model.py path/to/dataset/folder
Replace path/to/dataset/folder with the actual path containing the evaluation dataset.
The script will generate an output.txt file containing:

Trained model's architecture summary.
Evaluation metric(s) obtained (e.g., classification accuracy).
Additional insights or observations.
Requirements
Ensure you have the required Python packages installed. You can set up a virtual environment and install the dependencies using the following command:

bash
Copy code
virtualenv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
Evaluation
We evaluated the model's performance based on the provided dataset. The output.txt file contains detailed information about the model's architecture, evaluation metrics, and additional insights.

Conclusion
This project showcases our innovative approach to classifying sustainable apparel products using machine learning. By combining automated predictions with human expertise, we aim to enhance the accuracy and efficiency of the system. We believe this solution aligns with NITEX's vision for sustainable fashion and will contribute significantly to the future of the apparel industry.

For any further inquiries or assistance, please feel free to contact us.

Thank you for the opportunity to work on this transformative project.

Sincerely,
[Your Name]
[Your Email]
[Your Phone Number]
