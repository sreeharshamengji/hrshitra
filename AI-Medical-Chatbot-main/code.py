import re
import pandas as pd
import pyttsx3
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
import csv
import logging
from difflib import get_close_matches
import warnings
import os
print("Current working directory:", os.getcwd())
# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Helper Functions
def readn(nstr):
    """Text-to-Speech Function."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Use first available voice
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"Text-to-speech error: {e}")

def getSeverityDict():
    """Load Severity Data."""
    global severityDictionary
    try:
        with open('Symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    severityDictionary[row[0]] = int(row[1])
    except FileNotFoundError:
        logging.error("File 'Symptom_severity.csv' not found.")
        exit()

def getDescription():
    """Load Description Data."""
    global description_list
    try:
        with open('symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    description_list[row[0]] = row[1]
    except FileNotFoundError:
        logging.error("File 'symptom_Description.csv' not found.")
        exit()

def getPrecautionDict():
    """Load Precaution Data."""
    global precautionDictionary
    try:
        with open('symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 5:
                    precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    except FileNotFoundError:
        logging.error("File 'symptom_precaution.csv' not found.")
        exit()

def get_closest_match(symptom, symptoms_list):
    """Find the closest match for a symptom."""
    matches = get_close_matches(symptom, symptoms_list, n=1, cutoff=0.8)
    return matches[0] if matches else None

def tree_to_code(tree, feature_names):
    """Predict Disease Based on Symptoms."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    chk_dis = set(feature_names)
    symptoms_present = []

    while True:
        disease_input = input("Enter the symptom you are experiencing -> ").strip().replace(" ", "_")
        match = get_closest_match(disease_input, chk_dis)
        if match:
            disease_input = match
            print(f"Using closest match: {disease_input}")
            break
        print("Symptom not recognized. Please try again.")

    try:
        num_days = int(input(f"How many days have you been experiencing {disease_input}? "))
        if num_days <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 1 day.")
        num_days = 1

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease_idx = np.argmax(tree_.value[node])
            present_disease = tree.classes_[present_disease_idx]
            print(f"Predicted disease: {present_disease}")
            if present_disease in description_list:
                print(f"Description: {description_list[present_disease]}")
            if present_disease in precautionDictionary:
                print(f"Precautions: {', '.join(precautionDictionary[present_disease])}")
            else:
                print("Precautions not found for the predicted disease.")

    recurse(0, 1)

    # Print summary
    print("\n--- Diagnosis Summary ---")
    print(f"Symptoms provided: {', '.join(symptoms_present)}")
    print(f"Final Diagnosis: {present_disease}")
    if present_disease in precautionDictionary:
        print("Suggested Precautions:")
        for precaution in precautionDictionary[present_disease]:
            print(f"- {precaution}")

def getInfo():
    """Greet User and Collect Basic Info."""
    print("-----------------------------------AI Medical ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="-> ")
    name = input("Name: ")
    print(f"Hello, {name}! Let's begin diagnosing.")

# Main Entry Point
if __name__ == "__main__":
    # Load data
    try:
       training = pd.read_csv(r'C:\path\to\Training.csv').fillna(0)
       testing = pd.read_csv(r'C:\path\to\Testing.csv').fillna(0)

    except FileNotFoundError:
        logging.error("File 'Training.csv' or 'Testing.csv' not found.")
        exit()

    # Initialize dictionaries
    getSeverityDict()
    getDescription()
    getPrecautionDict()

    # Train model and start interaction
    clf = DecisionTreeClassifier().fit(training.iloc[:, :-1], training['prognosis'])
    getInfo()
    tree_to_code(clf, training.columns[:-1])

    print("----------------------------------------------------------------------------------------")
