# FashionMNIST
Image Classification on FashionMNIST Dataset with Data Analysis


## Introduction
This project aims to develop an AI solution for NITEX to identify and classify sustainable apparel products using the Fashion MNIST dataset. The solution includes data analysis, model development, and a human-in-the-loop approach to enhance accuracy. The project follows the guidelines provided and focuses on aligning with NITEX's vision for sustainable fashion.

## Project Structure
1. Analysis
2. Model Architecture & Training
3. Test Analysis
4. Human-in-the-Loop Approach
5. Requirements

   
### Analysis
*Analysis.ipynb: Jupyter Notebook for exploring the dataset and documenting insights.*

Findings: 
1. The raw csv contains the labels and the pixel intensity, The first column holds the label and next 784 column holds all the pixel value of the image.
2. The shape of the feature is 28x28x1, height and width of the images are 28x28 and has a channel 1 which indicates these images are grayscale
3. In the class distribution we can see, there are 10 labels (0->9) and each class contains 6,000 images making a total of 6000x10= 60,000 images.
4. The histogram on mean pixel intensity is left skewed (or negatively skewed) which indicates that the majority of the images in the dataset have lower mean pixel values, or they are relatively darker.
5. In the context of grayscale images, lower pixel values represent darker shades of gray, whereas higher pixel values represent lighter shades.
6. The histogram on standard deviation suggests that the pixel values within each image are relatively consistent and do not vary widely from the mean intensity.
7. I put a visualization for first 10 images of each label. According to the dataset descrioption the labels <ul>  <li>0 T-shirt/top </li> <li>1 Trouser</li> <li>2 Pullover </li> <li>3 Dress </li> <li>4 Coat </li><li>5 Sandal </li> <li>6 Shirt </li> <li>7 Sneaker </li> <li>8 Bag </li> <li>9 Ankle boot </li> </ul> Juidging by first 10 images, we can already tell the main mis-classification will occure between label 0 (T-shirt/top) and label 6 (6 Shirt), as they are both very similar, with very few details which could alsobe oversighted by humaneyes.

### Model Architecture & Training
*Training.ipynb: Jupyter Notebook for designing and training the machine learning model.*

#### Motivation: 
I tried to solve the problem with a custom CNN model, because I previously worked on a similar subject and personally I feel it is the best architecture suited for this types of problem. I started with building the model with two conv2d layers with filters=8. then added and removed BatchNormlization, Maxpool2d layers. First I kept the conv2d layer constant and changed the values and the frequencies for Maxpool2d and dropout layes, in order to find the best fit. After I was happy with the result where my model is in the bestfit (consistency between validation and training loss). I started to aim for the higher accuracies by increasing the conv2 layers. While I increased more conv2 layers, I had to change some values in the dropout layer to minimize the gap between training and validation loss. I kept a note on which layer is   contributing what to my architecture, which helped me find better results with the model. I also used two callback function to finetune the model, ReduceLROnPlateau decreases the learning rate when the model hits Plateau(not improving or deteriorating), then I used EarlyStopping which will stopp the epochs early if the model stars to deteriorate. I kept the patience in a manner so if ReduceLROnPlateau doesn't work it will stop by EarlyStopping. It helped me boost the accuracies to 0.9442 on the validation set. The validation is created by splitting the training set with .1 ratio. The final epoch contained the following data 
*loss: 0.0853 - accuracy: 0.9679 - val_loss: 0.1945 - val_accuracy: 0.9442*

### Test Analysis
*Testing.ipynb: Jupyter Notebook for testing my model and answer hypothesis*

In this notebook I evaluated my model with the testing set and achived 94.45% accuracy, then I printed a confusion matrix and a classification report. Confusion matrix helped me answer my first hypothesis which was the similarity pattern between label 0 and label 6. We can see from the confusion matrix the model indeed made mode errors predicting between these two labels. The classification report contains the overall evaluation metrices and per class evaluation metrices values. My model had a precision, f1 and recall value of .94, .94 & .94. 


### Human-in-the-Loop Approach
To enhance human-in-the-loop efficiency, I propose a method where human expertise can be combined with the model's predictions. Users can review uncertain predictions flagged by the model and provide corrections, improving the overall accuracy of the system. We can achive this by implement a confidence threshold for model predictions. Items with confidence scores below the threshold are flagged for human review. This selective approach optimizes human effort by focusing on uncertain predictions, maximizing efficiency. Develop an intuitive interface where human reviewers can access flagged items, view model predictions, and provide corrections or confirmations. 



### Model Evaluation
*evaluate_model.py: Script to evaluate the trained model on a given dataset folder.
output.txt: Text file generated after model evaluation, including model summary, evaluation metrics. For the insight, I added the confusion matrix to show-case what my model is predicting and where is it getting wrong.* 

To evaluate the trained model, run the evaluate_model.py script with the following command: The evalute_model.py file is in the Evaluation Script folder. While running the script DO NOT RUN  2> error.log command as it may interfere with the terminal prints. 
1. Make sure, my model and the script is in the same folder. 
2. The script will ask you if your dataset is in a csv file, if yes, make sure your csv is in the same format as the test dataset. So your csv will contain 785 columnds, where first column indicates the label. The script will automatically convert it to labels and features.
3. If your dataset is not in csv please but is in a numpy file, you can still continue
4. If your labels are not one-hot-encoded it's fine, you can enter the path, the script will convert it to one-hot-encoded for you
5. You may encounter some errors regarding mismatch shapes, this will occure when your number of labels do not match with the numbers of features.
6. If the dataset is not in either numpy or in csv, tough luck for me :(.



bash
```py
cd path/to/my/script-model-directory
python evaluate_model.py
```
Replace path/to/my/script-model-directory with the actual path containing the evaluation dataset.
The script will generate an output.txt file containing:


### Requirements
Ensure you have the required Python packages installed. You can set up a virtual environment and install the dependencies using the following command:
```py
virtualenv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

For any further inquiries or assistance, please feel free to contact me.
Sincerely,
Khan Abrar Shams
Khan.abrarshams@gmail.com
