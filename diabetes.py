

import streamlit
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

streamlit.title("Prediction for Diabetes")
data = pandas.read_csv("C:/Users/SAMSUNG/Desktop/pandas/Diabetes_stream/diabetes.csv")
print("done")

df = pandas.DataFrame(data)
print(df.isna().sum())

X = df.iloc[:,:-1]

y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


knn = KNeighborsClassifier()

model = knn.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred,y_test))

def input_features():
    
    streamlit.subheader("Enter Patient Details:")
    Pregnancies = streamlit.number_input("Pregnancies",0,20,1)
    Glucose = streamlit.number_input("Glucose Level",0,300,120)
    BloodPressure =  streamlit.number_input("Bloodpressure",0,122,70)
    SkinThickness = streamlit.number_input("SkinThickness",0,100,20)
    Insulin = streamlit.number_input("Insulin",0,900,80)
    BMI = streamlit.number_input("BMI",0,70,25)
    DiabetesPedigreeFunction  = streamlit.number_input(" DiabetesPedigreeFunction",0,3,1)
    Age = streamlit.number_input("Age",1,120,33)
     

    data = {
        "Pregnancies":Pregnancies,
        "Glucose":Glucose,
        "BloodPressure":BloodPressure,
        "SkinThickness":SkinThickness,
        "Insulin":Insulin,
        "BMI":BMI,
        "DiabetesPedigreeFunction":DiabetesPedigreeFunction,
        "Age":Age
    }
    
    
    features = pandas.DataFrame(data,index=[0])
    return features

input_df = input_features()

input_scaled = scaler.transform(input_df)

result = knn.predict(input_scaled)

if streamlit.button("show result"):
    
    if result == 1:
        
     streamlit.success(" The patient is Diabetic")
     
    else:
        
        streamlit.info(" The patient is Not Diabetic")