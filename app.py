import numpy as np
import pandas as pd
from flask import Flask, json, request, render_template
# from flask_pymongo import PyMongo
import pickle

app = Flask(__name__,  static_url_path = "/img", static_folder = "img")
# app.config["MONGO_URI"] = "mongodb://localhost:27017/CovidPrediction"
# mongo = PyMongo(app)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.values.to_dict()
    # print(data)
    orig_gender_arr = ['Gender_Female','Gender_Male','Gender_Transgender']
    orig_age_arr = ['Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60']
    orig_symptoms_arr = ['Fever','Tiredness','Dry-Cough','Difficulty-in-Breathing','Sore-Throat','None_Sympton','Pains','Nasal-Congestion','Runny-Nose','Diarrhea','None_Experiencing']
    orig_severity_arr = ['Severity_Mild','Severity_Moderate','Severity_None','Severity_Severe']
    orig_contact_arr = ['Contact_Dont-Know','Contact_No','Contact_Yes']
    patient_name = data['Obj[PatientName]']
    patient_gender_arr= []
    patient_severity_arr = []
    patient_age_arr = []
    patient_contacts_arr = []
    patient_gender_arr.append(data['Obj[gender]'])
    patient_severity_arr.append(data['Obj[severity]'])
    patient_age_arr.append(data['Obj[age]'])
    patient_contacts_arr.append(data['Obj[contact]'])
    patient_symptoms_arr = request.values.getlist('Obj[symptoms][]')

    def common(a, b):
        array1 = []
        for x in a:
            if x in b:
                array1.append(1)
            else:
                array1.append(0)

        return array1
    
    patient_symptoms = common(orig_symptoms_arr,patient_symptoms_arr)
    patient_age = common(orig_age_arr,patient_age_arr)
    patient_gender= common(orig_gender_arr,patient_gender_arr)
    patient_severity = common(orig_severity_arr,patient_severity_arr)
    patient_contacts = common(orig_contact_arr,patient_contacts_arr)
    
    final = patient_symptoms+patient_age+patient_gender+patient_severity+patient_contacts
    # print(patient_symptoms)
    # print(patient_age)
    # print(patient_gender)
    # print(patient_severity)
    # print(patient_contacts)
    # print(final)
    final_array = []
    final_array.append(final)
    if final.count(1) <= 6:
       outputt = "Negative"
    else:
       outputt = "Positive"      
    # print(final_array)
    # sample_dict = {"Fever": final[0], "Tiredness": final[1], "Dry-Cough": final[2], "Difficulty-in-Breathing" : final[3], "Sore-Throat" : final[4], "None_Sympton" : final[5], "Pains" : final[6], "Nasal-Congestion": final[7], "Runny-Nose": final[8], "Diarrhea": final[9], "None_Experiencing": final[10], "Age_0-9" : final[11], "Age_10-19": final[12], "Age_20-24": final[13], "Age_25-59": final[14], "Age_60": final[15], "Gender_Female": final[16], "Gender_Male": final[17], "Gender_Transgender": final[18], "Severity_Mild": final[19], "Severity_Moderate": final[20], "Severity_None": final[21], "Severity_Severe" : final[22], "Contact_Dont-Know" : final[23], "Contact_No" : final[24], "Contact_Yes" : final[25] }
    # final_dict = pd.DataFrame.from_dict(sample_dict)
    # print(final_dict)
    # int_features = [int(x) for x in final_dict]
    # final_features = [np.array(int_features)]

    prediction = model.predict(final_array)
    print(prediction[0])
    str1 = ""
    val = str1.join(prediction)
    # doc = mongo.db.predictions.insert_one({"patientName": patient_name,"patientAge": data['Obj[age]'],
    # "patientGender": data['Obj[gender]'],"patientSymptoms":patient_symptoms_arr,
    # "patientSeverity": data['Obj[severity]'], "patientContacts": data['Obj[contact]'],
    # "patientResult": val
    # })
    # print("Inserted") 

    if prediction[0] == "Negative":
        output ='Negative'
    elif prediction[0] == "Positive":
        output = 'Positive'
    
    return outputt
    # return render_template('index2.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
