import numpy as np
import pickle
import streamlit as st
import sklearn


# loading the saved model
loaded_model = pickle.load(open("trained_modell.sav", 'rb'))


# creating a function for Prediction

def heartdisease_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person does not have heart disease'
    else:
        return 'The person has heart disease'


def main():

    # giving a title
    st.title('CardiaRisk - Heart Disease Prediction web app')

    # getting the input data from the user

    age = st.text_input('Age')
    sex = st.text_input('Sex(Male=1, Female=0)')
    cp = st.text_input('Chest pain type(cp)')
    trestbps = st.text_input('Resting blood pressure(trestbps)')
    chol = st.text_input('Cholesterol')
    fbs = st.text_input('Fasting blood sugar(fbs)')
    restecg = st.text_input('Resting electrocardiographic results(restecg)')
    thalach = st.text_input('Maximum heart rate achieved( thalach)')
    exang = st.text_input('Exercise induced angina(exang)')
    oldpeak = st.text_input('Exercise relative to rest(oldpeak)')
    slope = st.text_input('Slope of the peak exercise ST segments')
    ca = st.text_input('Number of major vessels(ca)')
    thal = st.text_input('Thalassemia(thal)')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Disease Test Result'):
        try:
            # Convert input data to appropriate numeric types
            input_data = [
                int(age), int(sex), int(cp), int(trestbps), int(chol),
                int(fbs), int(restecg), int(thalach), int(exang),
                float(oldpeak), int(slope), int(ca), int(thal)
            ]
            diagnosis = heartdisease_prediction(input_data)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

    if diagnosis == 'The person does not have heart disease':
        st.success(diagnosis)
    elif diagnosis == 'The person has heart disease':
        st.error(diagnosis)

if __name__ == '__main__':
    main()
