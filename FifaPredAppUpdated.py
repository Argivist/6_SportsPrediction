import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# st.config.set_option('theme.base','auto')

cols = ['value_eur', 'release_clause_eur', 'age', 'potential',
        'movement_reactions']
model = pickle.load(open('./FifaDataPrediction.pickle', 'rb'))
data = pickle.load(open('./FifaScaler.pickle', 'rb'))
sd = data['sd']

scaler = data['sc']


def predict_rating(value_eur, release_clause_eur, age, potential, movement_reactions):
    input = pd.DataFrame(
        np.array([[value_eur, release_clause_eur, age, potential, movement_reactions]]).astype(np.float64))

    print(input)


    dummy_data = pd.DataFrame(data=np.zeros((input.shape[0], len(data['or']))), columns=data['or'])
    dummy_data[cols] = input
    scld = pd.DataFrame(scaler.transform(dummy_data), columns=dummy_data.columns)
    scld = scld[cols]
    # print(scld.columns)
    prediction = model.predict(scld)

    # getting the 95% confidence limit
    ci_upper_bound = prediction + 1.96 * sd
    ci_lower_bound = prediction - 1.96 * sd

    return prediction, ci_lower_bound, ci_upper_bound


def isnum(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Fifa Stats Prediction Model</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    movement_reactions = st.slider("movement_reactions", min_value=0, max_value=100, step=1)
    age = st.slider("age", min_value=0, max_value=100, step=1)
    potential = st.slider("potential", min_value=0, max_value=100, step=1)
    release_clause_eur = st.number_input("release clause (eur)", min_value=1, max_value=1000000000000000)
    value_eur = st.number_input("value (eur)", min_value=1, max_value=100000000000000)

   

    if st.button("Predict"):
        if isnum(release_clause_eur) and isnum(value_eur):
            output = predict_rating(value_eur, release_clause_eur, age, potential, movement_reactions)
            st.success('The predicted rating is {}'.format(output[0][0]))
            st.success('Confidence interval at 95% confidence: \n')
            st.success('lower limit: {}'.format(output[1][0]))
            st.success('upper limit: {}'.format(output[2][0]))
        else:
            st.error('Enter numeric values only in the textboxes')


if __name__ == '__main__':
    main()