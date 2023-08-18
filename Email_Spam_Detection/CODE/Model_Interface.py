# Modules:
import pandas as pd
import joblib as jb

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

E_spm_Model = jb.load("C:\\OneDrive\\Software\\Work_Spaces\\System_WorkSpace\\Data_WorkSpace\\Machine_Learning\\Email_Spam_Detection\\Models\\Email_Classification_Model.pkl")
print(type(E_spm_Model))
em ="C:\\OneDrive\\Software\\Work_Spaces\\System_WorkSpace\\Data_WorkSpace\\Machine_Learning\\Email_Spam_Detection\\Email.txt"
file = open(em,'r')
email=""
for  i in file:
    email+=i
#email = input("Enter the Email Text : ")

print(email)
vectorizer = CountVectorizer()

X_test_vectorized = vectorizer.transform([email])

Prediction = E_spm_Model.predict(X_test_vectorized)

print(Prediction)