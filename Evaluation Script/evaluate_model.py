# import io
# import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def welcome():
    print('Welcome! You can quit anytime by pressing (e)')
    script()

def goodbye():
    print('\n\nGoodBye!')


def accuracy(x_test,y_test,model):
    print("\nCalculating Accuracy: ")
    
    try:
        predictions = model.predict(x_test)
        predicted_labels = predictions.argmax(axis=1)
        true_labels = y_test.argmax(axis=1)
        accuracy = accuracy_score(true_labels, predicted_labels)

        with open('output.txt', 'a') as testwritefile:
            testwritefile.write("\n")
            testwritefile.write("ACCURACY ON YOUR DATA: "+ str(accuracy) + '\n')
            testwritefile.write("\n")
            testwritefile.write("_________________________________________________________________\n")
            testwritefile.close()
        print("[Done]")
        confusion_mat(true_labels, predicted_labels)
    
    except ValueError as error:
        print("[Failed]")
        print("We encountered a shape mismatch")
        print("Please make sure you are not putting the feature files on the labels and vise versa")
    
    except Exception as error:
        print("[Failed]")
        print("problem with calculating accuracy")
        print(error)
    

def classification_rep(true_y, predicted_y):
    print("\nCreating Classification Report: ", end='')
    
    try:
        classification_res = classification_report(true_y, predicted_y)
        with open('output.txt', 'a') as testwritefile:
            testwritefile.write("Classification Report\n")
            testwritefile.write("-----------------------------------------------------------------\n")            
            testwritefile.write(classification_res)
            testwritefile.close
            
        print("[Done]")
        goodbye()
        
    except Exception as error:
        print("Error while creating classification report")
        print(error)
    

def confusion_mat(true_y, predicted_y):
    
    print("\nCreating Confusion Matrix: ", end='')
    
    try:
        confusion_res = confusion_matrix(true_y, predicted_y)                
        with open('output.txt', 'a') as testwritefile:
            testwritefile.write("Confusion Matrix\n")
            testwritefile.write("-----------------------------------------------------------------\n")
            
            for row in confusion_res: #this loop to only beautify the confusion matrix
                formatted_row = " ".join("{:<5}".format(cell) for cell in row)
                testwritefile.write(formatted_row+'\n')
            testwritefile.write("_________________________________________________________________\n")
            testwritefile.close
        print("[DONE]")
        classification_rep(true_y, predicted_y)
          
    except Exception as error:
        print("Error while calculating Confusion Matrix")
        print(error)
    
    
def script():
    ans = input("Is your Dataset on a CSV file? (y/n)\n")
    if (ans.lower()=='y'):
        print("\n\nPlease make sure, you csv is in the same format as the dataset [labels following by 748pxl values]")
        csv_dataset()

    elif(ans.lower()=='n'):
        no_csv_dataset()
    
    elif(ans.lower()=='e'):
        goodbye()
    
    else:
        print('\n\nPlease provide a valid input')
        script()

        
def csv_dataset():
    loc = input('\nPlease enter the file path to your CSV: ')
    
    if loc.lower()=='e':
        goodbye()
    
    else:
        try:
            test_dataset = pd.read_csv(loc,encoding='ISO-8859-1')
            print("\nFound your Dataset! Hold on a bit while we process your data: ", end='')
            x, y = data_prep(test_dataset)
            print("[Done]")
            loadmodel(x,y)
            
        except OSError as error:
            print("Can't find the file, Please enter the correct path. Make sure to write the extension too :(\n")
            csv_dataset()

        except Exception as error:
            print("\nOpps! looks like we encountered a problem.", end='')
            print(error)
            csv_dataset()
           
        
def no_csv_dataset():
    try:
        query = str(input("\nDo you have a separate npy file for label and images?(y/n/e)\n"))
        
        if query.lower()=='y':
            query = str(input("Are your labels one-hot-encoded?(y/n/e)\n"))
            if query.lower()=='y':
                no_csv_dataloader()
                
                
            elif query.lower()=='n':
                one_hot_encoder()
                
                
            elif query.lower()=='e':
                goodbye()
            else:
                no_csv_dataset()
            
        elif query.lower()=='n':
            print('Sorry! We can\'t help you :( I only accept npy for csv')
        
        elif query.lower()=='e':
            goodbye()
    
    except Exception as error:
        print(error)

        
def one_hot_encoder():
    labels_location = str(input('\nEnter the path to your labels.npy: '))
    no_csv_dataloader(labels_location)

        
def no_csv_dataloader(labels_location = ''):
    feature_location = str(input('\nEnter the path to your images.npy: '))
    
    try:
        feature = np.load(feature_location)
    
    except Exception as error:
        print("\nThere was a problem finding your images.npy")
        print(error)
    
    
    if labels_location == '':
        labels_location = str(input('\nEnter the path to your labels.npy: '))
        try:
            labels = np.load(labels_location)
            loadmodel(feature, labels)
        except Exception as error:
            print("\nThere was a problem finding your labels.npy")
            print("\n1. Make sure you are entering the path correctly")
            print("\n2. Make sure you are adding the .npy extension in the input")

    else:
        try:
            labels = np.load(labels_location)
            labels_one_hot = to_categorical(labels, 10)
            print("\nYour labels are now one-hot-encoded")
            loadmodel(feature, labels_one_hot)
        except Exception as error:
            print(error)

        
        
        
def loadmodel(x, y):
    print("\nSearching For Model: ", end="")
    try:
        model = load_model('MNISTFASHION.h5')
        print("[Model Found]")
        
        try:
            with open('output.txt', 'a') as file:
                print("Model Summary:", file=file)
                model.summary(print_fn=lambda x: file.write(x + '\n'))
                file.close()

            accuracy(x, y, model)
            
        except Exception as error:
            print(error)
              
    except OSError as error:
        print('\n\nMODEL not found.')
        print('1. Make sure your model is in the same folder.')
        print('2. Make sure your model\'s name is set to MNISTFASHION.h5.')
        print('3. Make sure it is an .h5 file. ', end='\nOriginal Error Message: ')
        print(error)
        
    except Exception as error:
        print('\n\nAn error occurred while loading the model:')
        print(error)

        
def data_prep(csv_data):
    img_height, img_width = 28, 28
    num_classes = 10
    channel = 1
    labels_one_hot = to_categorical(csv_data.label, num_classes)
    num_images = csv_data.shape[0]
    features_x = csv_data.values[:, 1:]
    img_feature = features_x.reshape(num_images, img_height, img_width, channel)
    features_x = img_feature / 255 
    return features_x, labels_one_hot
    
welcome()