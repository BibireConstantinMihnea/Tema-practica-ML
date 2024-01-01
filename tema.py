import numpy as np
import os

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

