import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class NaiveBayesClassifier:
    def encode_class(self, mydata):
        classes = []
        for i in range(len(mydata)):
            if mydata[i][-1] not in classes:
                classes.append(mydata[i][-1])
        for i in range(len(classes)):
            for j in range(len(mydata)):
                if mydata[j][-1] == classes[i]:
                    mydata[j][-1] = i
        return mydata

    def splitting(self, mydata, ratio):
        train_num = int(len(mydata) * ratio)
        train = []
        test = list(mydata)
        while len(train) < train_num:
            index = random.randrange(len(test))
            train.append(test.pop(index))
        return train, test

    def groupUnderClass(self, mydata):
        data_dict = {}
        for i in range(len(mydata)):
            if mydata[i][-1] not in data_dict:
                data_dict[mydata[i][-1]] = []
            data_dict[mydata[i][-1]].append(mydata[i])
        return data_dict

    def MeanAndStdDev(self, numbers):
        avg = np.mean(numbers)
        stddev = np.std(numbers)
        return avg, stddev

    def MeanAndStdDevForClass(self, mydata):
        info = {}
        data_dict = self.groupUnderClass(mydata)
        for classValue, instances in data_dict.items():
            numerical_values = [float(attribute[0]) for attribute in instances if attribute[0].replace('.', '', 1).isdigit()]
            info[classValue] = [self.MeanAndStdDev(numerical_values)][0] if numerical_values else (0, 0)
        return info

    def calculateGaussianProbability(self, x, mean, stdev):
        epsilon = 1e-10
        expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
        return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

    def calculateClassProbabilities(self, info, test):
        probabilities = {}
        for classValue, classSummaries in info.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, std_dev = classSummaries[i]
                x = test[i]
                probabilities[classValue] *= self.calculateGaussianProbability(x, mean, std_dev)
        return probabilities

    def predict(self, info, test):
        probabilities = self.calculateClassProbabilities(info, test)
        bestLabel = max(probabilities, key=probabilities.get)
        return bestLabel

    def getPredictions(self, info, test):
        predictions = [self.predict(info, instance) for instance in test]
        return predictions

    def accuracy_rate(self, test, predictions):
        correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
        return (correct / float(len(test))) * 100.0

def load_data(part_folder_path):
    emails = []
    labels = []
    if os.path.isdir(part_folder_path):
        for filename in os.listdir(part_folder_path):
            is_spam = filename.startswith('spm')
            file_path = os.path.join(part_folder_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                emails.append(content)
                labels.append(1 if is_spam else 0)
    return emails, labels
dataset_path = 'F:\\ML\\lingspam_public'

training_emails, training_labels = [], []
testing_emails, testing_labels = [], []

for main_folder in os.listdir(dataset_path):
    main_folder_path = os.path.join(dataset_path, main_folder)

    if os.path.isdir(main_folder_path):
        for part_num in range(1, 10):
            part_path = os.path.join(main_folder_path, f'part{part_num}')
            part_emails, part_labels = load_data(part_path)
            training_emails += part_emails
            training_labels += part_labels

        p10_emails, p10_labels = load_data(os.path.join(main_folder_path, 'part10'))
        testing_emails += p10_emails
        testing_labels += p10_labels

mydata = []
for email, label in zip(training_emails, training_labels):
    mydata.append([email, label])  # Using lists instead of tuples

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_emails)
X_test = vectorizer.transform(testing_emails)


nb_classifier = NaiveBayesClassifier()

mydata_encoded = NaiveBayesClassifier().encode_class(mydata)
train_data, test_data = NaiveBayesClassifier().splitting(mydata_encoded, 0.8)

# Prepare training and testing emails/labels
training_emails = [data[0] for data in train_data]
training_labels = [data[-1] for data in train_data]

testing_emails = [data[0] for data in test_data]
testing_labels = [data[-1] for data in test_data]

# Vectorization using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_emails)
X_test = vectorizer.transform(testing_emails)

# Model training with your Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier()
info = nb_classifier.MeanAndStdDevForClass(train_data)

# Fit and predict using sklearn's model
model = SVC()
model.fit(X.toarray(), np.array(training_labels))
predicted = model.predict(X_test.toarray())

# Calculate and print accuracy
accuracy = accuracy_score(testing_labels, predicted)
print(f"Accuracy: {accuracy}")

# Creating a confusion matrix and displaying it
cm = confusion_matrix(testing_labels, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot()
plt.show()