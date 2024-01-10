import os
import re
from collections import Counter

def load_data(path):
    emails = []
    labels = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            is_spam = filename.startswith('spm')
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                emails.append(content)
                labels.append(1 if is_spam else 0)
    return emails, labels

def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def preprocess(emails):
    processed = []
    for email in emails:
        words = tokenize(email)
        processed.append(words)
    return processed

def calc_prior(labels):
    total = len(labels)
    spam_count = sum(labels)
    clean_count = total - spam_count
    prior_spam = spam_count / total
    prior_clean = clean_count / total
    return prior_spam, prior_clean

def calc_likelihood(emails, labels):
    likely_spam = {}
    likely_clean = {}
    spam_count = Counter()
    clean_count = Counter()

    for i, email in enumerate(emails):
        label = labels[i]
        for word in email:
            if label == 1:
                spam_count[word] += 1
            else:
                clean_count[word] += 1

    total_spam = sum(spam_count.values())
    total_clean = sum(clean_count.values())

    for word, count in spam_count.items():
        likely_spam[word] = count/total_spam

    for word, count in clean_count.items():
        likely_clean[word] = count/total_clean

    return likely_spam, likely_clean

def naive_bayes_classifier(email, prior_spam, prior_clean, likely_spam, likely_clean):
    spam_score = prior_spam
    clean_score = prior_clean

    for word in email:
        spam_score *= likely_spam.get(word, 1e-10)
        clean_score *= likely_clean.get(word, 1e-10)

    if spam_score > clean_score:
        return 1
    else: return 0

def evaluation(training_emails, training_labels, testing_emails, testing_labels):
    training_emails = preprocess(training_emails)
    testing_emails = preprocess(testing_emails)
    prior_spam, prior_clean = calc_prior(training_labels)
    likely_spam, likely_clean = calc_likelihood(training_emails, training_labels)

    predictions = []
    corrrect_pred = 0
    for email in testing_emails:
        predictions.append(naive_bayes_classifier(email, prior_spam, prior_clean, likely_spam, likely_clean))

    for pred, label in zip(predictions, testing_labels):
        if pred == label:
            corrrect_pred += 1
    accuracy = corrrect_pred / len(testing_labels)
    print(f"Accuracy: {accuracy:.4%}")

def LOOCV(training_emails, training_labels):
    correct_pred = 0
    for i in range(len(training_emails)):
        loocv_email = training_emails[i]
        loocv_label = training_labels[i]

        training_set_emails = training_emails[:i] + training_emails[i+1:]
        training_set_labels = training_labels[:i] + training_labels[i+1:]

        training_set_emails = preprocess(training_set_emails)
        prior_spam, prior_clean = calc_prior(training_set_labels)
        likely_spam, likely_clean = calc_likelihood(training_set_emails, training_set_labels)

        prediction = naive_bayes_classifier(loocv_email, prior_spam, prior_clean, likely_spam, likely_clean)

        if prediction == loocv_label:
            correct_pred += 1

    accuracy = correct_pred / len(training_labels)
    print(f"LOOCV Accuracy: {accuracy:.4%}")

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

evaluation(training_emails, training_labels, testing_emails, testing_labels)
#LOOCV(testing_emails, testing_labels)