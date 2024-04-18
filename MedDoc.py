import pandas as pd
from sklearn.model_selection import train_test_split

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig
from sklearn.model_selection import KFold



###########################
#setting seeds for random operaations to occur
import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)

# Parsing the dataset category
import re

'''def extract_features(data):
   
   extracted_features = data.drop(columns=["fbs", "exang", "oldpeak", "slope", "ca", "thal"], errors="ignore")
    
   return extracted_features'''


def preprocess_features(features):
    # Drop the unwanted columns
    preprocessed_features = features.drop(columns=["fbs", "exang", "oldpeak", "slope", "ca", "thal"], errors="ignore")
    
    # Skip the first row (column header) and reset the index
    preprocessed_features = preprocessed_features.iloc[1:].reset_index(drop=True)
    
    return preprocessed_features

def extract_labels(data):
    labels = data.iloc[:, -1]  # Select the last column as labels
    labels = labels.iloc[1:]  # Exclude the first row (column header)
    labels = labels.replace('', np.nan).dropna().astype(int)  # Convert to numeric data type
    return labels


data = pd.read_csv('data2.csv', header=None )  # Skip the first row

labels = extract_labels(data)
print(labels)

labels = labels.to_numpy()



def preprocess_labels(labels):
    return labels

# Load the data from data.csv
file_path = "data2.csv"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip', names=column_names)




# Print the column names to check for the presence of the "num" column
print(data.columns)


# Extract and preprocess features
'''features = extract_features(data)'''
preprocessed_features = preprocess_features(data)

# Extract and preprocess labels
labels = extract_labels(data)
print(labels.dtype)
print(np.unique(labels))
preprocessed_labels = preprocess_labels(labels)

###########################
#Building the model

##loading the data sub category
def read_file(file_path):
    
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    
    data_cleaned = data.dropna()  # Remove rows with missing values
    return data_cleaned

def encode_categorical_variables(data):
    
    data_encoded = pd.get_dummies(data)  # One-hot encode categorical variables
    return data_encoded

def split_train_test(data, test_size=0.2, random_state=42):
    """
    Split the data into training and testing subsets.
    
    Parameters:
    - data (pandas.DataFrame): The dataset to be split.
    - test_size (float): The proportion of the data to be used for testing.
    - random_state (int): The random seed for reproducibility.
    
    Returns:
    - X_train (pandas.DataFrame): The training features.
    - X_test (pandas.DataFrame): The testing features.
    - y_train (pandas.Series): The training labels.
    - y_test (pandas.Series): The testing labels.
    """
    features = data.drop(columns=["num"])  # num is the label column(what I'm trying to predict)
    labels = data["num"]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
 

# Clean Data

def remove_duplicates(data):
    
    cleaned_data = data.drop_duplicates()
    return cleaned_data

def remove_outliers(data):
    numeric_columns = data.select_dtypes(include=np.number).columns

    # Select only the numeric columns
    numeric_data = data[numeric_columns]

    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1
    threshold = 1.5  # Adjust the threshold based on your dataset

    # Remove rows with outliers
    cleaned_data = data[~((numeric_data < (q1 - threshold * iqr)) | (numeric_data > (q3 + threshold * iqr))).any(axis=1)]
    return cleaned_data


def handle_noise(data, window_size=10):
    # Handle missing values
    data = data.replace('?', np.nan)

    # Convert non-numeric columns to numeric
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Apply rolling mean
    smoothed_data = data.rolling(window_size, center=True).mean()
    return smoothed_data


print(data['num'].unique())



def handle_imbalance(data):
    features = data.iloc[1:, :-1]  # Exclude the first row (column names) and last column (labels)
    labels = data.iloc[1:, -1]  # Include only the last column (labels)

    # Replace '?' with NaN in features
    features = features.replace('?', np.nan)

    # Convert the features to numpy array
    X = features.values

    # Convert the labels to binary classes (0 and 1)
    labels = labels.replace({'?': np.nan, '0': 0, '1': 1})

    # Replace non-finite values (NaN or inf) in labels with -1
    labels = labels.replace([np.nan, np.inf, -np.inf], -1)

    # Replace NaN values in labels with the most frequent class
    imputer = SimpleImputer(strategy='most_frequent')
    y = imputer.fit_transform(labels.values.reshape(-1, 1)).flatten()

    # Convert the labels to integers
    y = y.astype(int)

    # Perform random undersampling
    undersampler = RandomUnderSampler(random_state=42)
    x_undersampled, y_undersampled = undersampler.fit_resample(X, y)

    # Perform random oversampling
    oversampler = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = oversampler.fit_resample(x_undersampled, y_undersampled)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)

    # Create a dictionary to store the resampled data and class weights
    resampled_data = {
        'x_resampled': x_resampled,
        'y_resampled': y_resampled,
        'x_undersampled': x_undersampled,
        'y_undersampled': y_undersampled,
        'class_weights': class_weights
    }

    return resampled_data


# Load the dataset
data = pd.read_csv('data2.csv')

# Handle class imbalance
resampled_data = handle_imbalance(data)

# Access the resampled data
x_resampled = resampled_data['x_resampled']
y_resampled = resampled_data['y_resampled']
x_undersampled = resampled_data['x_undersampled']
y_undersampled = resampled_data['y_undersampled']
class_weights = resampled_data['class_weights']





# Load and preprocess data2.csv
data2 = pd.read_csv('data2.csv')

# Ensure labels are correctly encoded (0s and 1s)
data2['num'] = data2['num'].astype(int)

# Split the data into training, validation, and test sets
train_data, val_test_data, train_labels, val_test_labels = train_test_split(
    data2.drop(columns=['num']),  # Features
    data2['num'],  # Labels
    test_size=0.3,  # Adjust the split ratio as needed
    random_state=42
)

# Split the test set into validation and test sets
val_data, test_data, val_labels, test_labels = train_test_split(
    val_test_data,
    val_test_labels,
    test_size=0.5,  # Split the remaining data into validation and test
    random_state=42
)

# Replace the above data splits with the preprocessed data
train_data = x_resampled
train_labels = y_resampled
val_data = x_resampled  # You can use the same resampled data for validation
val_labels = y_resampled


print(val_data)
#Replacing NaN vaalues first
import numpy as np

# Iterate over the columns and replace NaN values with 0
for column in range(val_data.shape[1]):
    val_data[:, column] = np.nan_to_num(val_data[:, column], nan=0)

# Replace '?' with NaN
test_data = test_data.replace('?', np.nan)

import pandas as pd

# Conversion to pandas DataFrame
val_data = pd.DataFrame(val_data, columns=data2.columns[:-1])
train_data = pd.DataFrame(train_data, columns=data2.columns[:-1])
test_data = pd.DataFrame(test_data, columns=data2.columns[:-1])

# Fill NaN values with 0 in the entire DataFrame
val_data = val_data.fillna(0)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# Convert the entire DataFrame to integers
val_data = val_data.astype(int)
train_data = train_data.astype(int)
test_data = test_data.astype(int)


test_labels = test_labels.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)






import spacy
# Load the pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')


# Define mappings for the categorical variables
sex_mapping = {"male": 0, "female": 1}
cp_mapping = {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4}
chol_mapping = {"normal": 0, "borderline high": 1, "high": 2}


# function to preprocess the input text
def preprocess_input(input_text):
    # Use spaCy to tokenize and process the input text
    doc = nlp(input_text.lower())
    
    # Initialize default values for variables
    age = None
    sex = None
    cp = None
    trestbps = None
    chol = None
    
    # Extract information from the processed text
    for token in doc:
        if token.like_num:
            # If a numerical value is detected, assume it's the age
            age = int(token.text)
        elif token.text in sex_mapping:
            sex = sex_mapping[token.text]
        elif token.text in cp_mapping:
            cp = cp_mapping[token.text]
        elif token.text in chol_mapping:
            chol = chol_mapping[token.text]
    
    # Return the preprocessed values as a dictionary
    return {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol}

#bert_classifier function

class BertClassifier(nn.Module):
    def __init__(self, bert, classifier):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Check if the data at this index is not empty
        if isinstance(self.data, dict):
            # If self.data is a dictionary, access data using keys
            data_row = self.data['text_data'][idx]
        else:
            # If self.data is not a dictionary, assume it's a DataFrame
            data_row = self.data.iloc[idx].values

        # Handle missing or problematic data here
        if np.any(pd.isnull(data_row)) or np.any(data_row == '?'):
            data_row = np.zeros_like(data_row, dtype=np.float32)

        sample = {
            'text_data': torch.tensor(data_row, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample


def train_model(train_data, train_labels, val_data, val_labels, test_data, test_labels, labels):
 # custom dataset class used for training validation and testing
 import torch
 from torch.utils.data import Dataset
 import numpy as np



 # Reseting the index of the DataFrame
 test_data.reset_index(drop=True, inplace=True)

 # Create custom datasets for training, validation, and testing
 train_dataset = CustomDataset(train_data, train_labels)
 val_dataset = CustomDataset(val_data, val_labels)
 test_dataset = CustomDataset(test_data, test_labels)


 #definition of terms used below
 learning_rate = 0.001
 num_epochs = 10
 early_stopping_patience = 5

 # Define batch size
 batch_size = 32  # can adjust as needed

 # Create data loaders
 from torch.utils.data import DataLoader
 train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
 test_data_loader = DataLoader(test_dataset, batch_size=batch_size)


 # Import necessary libraries
 import torch
 import torch.nn as nn
 import torch.optim as optim
 from torch.utils.data import DataLoader
 from transformers import BertModel, BertConfig

 # Model Definition (Bert-based model)
 config = BertConfig.from_pretrained("bert-base-uncased")
 bert_model = BertModel.from_pretrained("bert-base-uncased", config=config)

 num_classes = 2  # Change this to the number of classes in your classification task
 classifier = nn.Sequential(
     nn.Linear(config.hidden_size, 256),
     nn.ReLU(),
     nn.Dropout(0.1),
     nn.Linear(256, num_classes)
 )

 model = BertClassifier(bert_model, classifier)

 # Check if a GPU is available and set the device accordingly
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 # Setting requires_grad to True for model parameters
 for param in model.parameters():
     param.requires_grad = True

 # Move the model to the device
 model = model.to(device)

 # Define the loss function (CrossEntropyLoss for classification)
 criterion = nn.CrossEntropyLoss()

 # Define the optimizer (Adam optimizer)
 learning_rate = 0.001  # Adjust as needed
 optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 #plotting training and validation loss curves
 import matplotlib.pyplot as plt

 #labels-series
 labels_series = data['num']

 # Training Loop
 num_epochs = 10  # Adjust as needed
 best_validation_loss = float("inf")
 early_stopping_patience = 3  # Adjust as needed
 early_stopping_counter = 0

 # Lists to store losses
 train_losses = []
 val_losses = []

 for epoch in range(num_epochs):
     model.train()
     total_train_loss = 0.0

     # Cross-validation using folds which splits data into multiple subsets.
     from sklearn.model_selection import KFold

     # Define the number of splits (folds)
     num_splits = 5  # Adjust as needed

     kf = KFold(n_splits=5, random_state=42, shuffle=True)

     # Lists to store validation losses for each fold
     fold_val_losses = []

     for fold, (train_index, val_index) in enumerate(kf.split(preprocessed_features.reset_index(drop=True), labels_series)):
         print("Labels shape:", labels.shape)
         print("Preprocessed Features shape:", preprocessed_features.shape)

         X_train, X_val = preprocessed_features.iloc[train_index], preprocessed_features.iloc[val_index]
         y_train, y_val = labels_series.iloc[train_index], labels_series.iloc[val_index]

         # Training loop
         for batch in train_data_loader:
             inputs = batch['text_data']
             attention_mask = (inputs != 0)

             # Convert inputs to a LongTensor if it's not already
             if inputs.dtype != torch.long:
                 inputs = inputs.long()

             inputs = inputs.to(device)
             attention_mask = attention_mask.to(device)

             labels = batch['label']
             optimizer.zero_grad()
             outputs = model(input_ids=inputs, attention_mask=attention_mask)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()
             total_train_loss += loss.item()

         avg_train_loss = total_train_loss / len(train_data_loader)
         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

         # Validation loop
         model.eval()
         total_val_loss = 0.0

         with torch.no_grad():
             for batch in val_data_loader:
                 inputs = batch['text_data']
                 attention_mask = (inputs != 0)

                 # Convert inputs to a LongTensor if it's not already
                 if inputs.dtype != torch.long:
                     inputs = inputs.long()

                 inputs = inputs.to(device)
                 attention_mask = attention_mask.to(device)

                 labels = batch['label']
                 outputs = model(inputs, attention_mask=attention_mask)
                 loss = criterion(outputs, labels)
                 total_val_loss += loss.item()

         avg_val_loss = total_val_loss / len(val_data_loader)
         fold_val_losses.append(avg_val_loss)
         print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

     # Append the average validation loss for this epoch
     val_losses.append(sum(fold_val_losses) / len(fold_val_losses))

     # Append the average training loss for this epoch
     train_losses.append(avg_train_loss)

 # Now plot the training and validation losses outside the loop
 plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
 plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.legend()
 plt.show()


 # Testing Loop
 model.eval()
 total_test_loss = 0.0
 correct_predictions = 0
 total_predictions = 0

 with torch.no_grad():
     for batch in test_data_loader:
         inputs = batch['text_data']
         input_ids = inputs.to(torch.long)
         inputs = inputs.to(device)

         labels = batch['label']
         outputs = model(input_ids=input_ids, attention_mask=(inputs != 0))
         loss = criterion(outputs, labels)
         total_test_loss += loss.item()

         # Calculate accuracy
         _, predicted = torch.max(outputs, 1)
         correct_predictions += (predicted == labels).sum().item()
         total_predictions += labels.size(0)

 # Calculate average test loss
 avg_test_loss = total_test_loss / len(test_data_loader)
 print(f"Test Loss: {avg_test_loss:.4f}")

 # Calculate accuracy
 accuracy = correct_predictions / total_predictions
 print(f"Accuracy: {accuracy * 100:.2f}%")

 # Save the model
 torch.save(model.state_dict(), 'model.pth')
 

 # Load the model
 model.load_state_dict(torch.load('model.pth'))
 model.eval()
 return model


# Example usage:
user_input = "I am a 45-year-old male with typical angina and high cholesterol"
processed_input = preprocess_input(user_input)
print(processed_input)

#passing input for processing by preloaded model
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize and encode the preprocessed user input
input_text = "I am a 45-year-old male with typical angina and high cholesterol"
encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

# Pass the input through the BERT model
outputs = bert_model(**encoded_input)

# Extract the pooled output (CLS token) or the hidden states as per your requirement
pooled_output = outputs.pooler_output  # Example: using the pooled output

# Now you can pass the pooled_output to your classifier for prediction

# Define your classifier
classifier = nn.Sequential(
    nn.Linear(768, 256),  # Assuming BERT pooled_output size is 768
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2)  # Assuming you have 2 output classes
)

# Pass the pooled_output to the classifier for prediction
logits = classifier(pooled_output)

# Apply softmax to obtain probabilities
probs = nn.functional.softmax(logits, dim=-1)

# Obtain the predicted class
predicted_class = torch.argmax(probs)

# Convert the class to 1 or 0 as per your requirement
if predicted_class == 1:
    output = 1
else:
    output = 0

print(output)  # This would be the predicted output (either 1 or 0)

''''Training: The train_model function takes the training data (data2.csv), processes it, and trains a model to learn patterns and associations between input features and labels.
Prediction: The provided snippet processes a user input, passes it through the trained model (BERT + classifier), and produces a prediction (either 1 or 0) based on what the model has learned during training.
'''

'''#neeeewwwwww
# Load the saved model (assuming it's 'model.pth')
model = torch.load("model.pth")
print(type(model))

# Set the model to evaluation mode (optional but recommended for prediction)
#model.eval()

# Example: Assuming model expects 'age', 'sex', etc. features directly
prediction = model(processed_input)

# Process the prediction based on your model's output (logits, probabilities, etc.)'''
