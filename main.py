import streamlit as st
import pandas as pd
import altair as alt

from urllib.error import URLError
import pandas as pd
import csv
import pickle
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

file = open('final_dataset.csv')
csv_file = csv.reader(file)

symptom_file = open('archive/Symptom-severity.csv')
symptom_csv = csv.reader(symptom_file)

precautions = open('archive/symptom_precaution.csv')
prec_csv = csv.reader(precautions)

header = next(csv_file)
symptoms_header = next(symptom_csv)

symptom_hasher = {}
count = 0

for symptom in symptom_csv:
    symptom_hasher[symptom[0]] = count
    count += 1

rows = []

for row in csv_file:
    rows.append(row)

disease_symptoms = []

for row in rows:
    x = []
    x.append(row[0])

    for i in range(17):
        if row[i + 1] != '':
            a = row[i + 1].strip() # preceeding and succeeding whitespaces removal
            b = a.split(" ") # the variables b and c achieve removal of whitespaces in between.
            c = ''.join(b)
            x.append(symptom_hasher[c])

    disease_symptoms.append(x)


final_data = []

for disease in disease_symptoms:
  symptom_indices = disease[1:]
  i = len(symptom_indices)
  zeros = [0] * 133
  for ii in range(i):
    zeros[symptom_indices[ii]] = 1
  vector = [disease[0]] + zeros
  final_data.append(vector)



df = pd.DataFrame(final_data,columns = ['Disease'] + [i for i in range(133)])



clean_data = df.drop(117,axis='columns')

inputs = clean_data.drop("Disease",axis='columns')
target = clean_data["Disease"]


le_disease = LabelEncoder()

target['Disease_final'] = le_disease.fit_transform(target)



final_target = pd.DataFrame(target["Disease_final"],columns=["Disease"])



X_train,X_test,Y_train,Y_test = train_test_split(inputs,final_target, test_size= 0.85)

model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

model.score(X_test,Y_test)






st.set_page_config(page_title='Medical Diagnosis App', page_icon = "🏥")

@st.cache

def predict_disease(inp):
    return le_disease.inverse_transform(model.predict([inp]))

def get_symptoms():
    df_simp = pd.DataFrame(list(symptom_hasher.keys()),columns=['Symptoms'])
    return df_simp

try:
    df = get_symptoms()
    input_Simps = st.multiselect(
        "Choose Symptoms", list(df['Symptoms'])
    )
    if not input_Simps:
        st.success("No symptoms? Hooray! You're healthy...")
    else:
        inp = [0]*132
        for simp in input_Simps:
            inp[symptom_hasher[simp]] = 1

        x = predict_disease(inp)[0]
        st.write(f'The predicted Disease is: {x}')

        message = ""

        for disease in prec_csv:
            if disease[0].lower() == x.lower():
                out = "You may want to follow these precautions : \n"
                for precaution in disease[1:]:
                    if precaution != "":
                        out = out + precaution.capitalize() + ", \n"

                message = out
                break

        st.info(message)



except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
