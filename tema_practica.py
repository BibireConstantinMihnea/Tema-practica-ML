
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_emails)
X_test = vectorizer.transform(testing_emails)


X_train, X_validation, y_train, y_validation = train_test_split(X, training_labels, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


predicted = model.predict(X_validation)


cm = confusion_matrix(y_validation, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])


disp.plot()
plt.show()
