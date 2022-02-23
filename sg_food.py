

import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
import os
import altair as alt
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
import tensorflow_hub as hub
from utils import *

st.set_page_config(page_title="SG Snap Food",
                   page_icon="🍔")

labels = {0: 'Hainanese Chicken Rice', 1: 'apple', 2: 'bak chor mee',  3: 'bak kut teh',
          4: 'ban mian soup', 5: 'banana', 6: 'char kway teow', 7: 'chendol',
          8: 'chicken curry noodle', 9: 'claypot rice',  10: 'curry puff',
          11: 'fish head curry', 12: 'fried carrot cake', 13: 'grapes', 14: 'ice kacang',
          15: 'kiwi', 16: 'laksa', 17: 'mango', 18: 'mee hoon kueh', 19: 'mee rebus', 20: 'mee siam',
          21: 'nasi briyani', 22: 'nasi lemak', 23: 'orange', 24: 'oyster omelette', 25: 'pear',
          26: 'pineapple', 27: 'pomegranate', 28: 'popiah', 29: 'prawn noodles soup',
          30: 'roti john',  31: 'roti prata', 32: 'satay', 33: 'wanton mee', 34:'watermelon'}

sgfood = ['Hainanese chicken rice', 'Bak chor mee', 'Bak kut teh', 'Ban mian soup', 'Char kway teow', 'Chendol', 'Chicken curry noodle',
          'Claypot rice', 'Curry puff', 'Fish head curry', 'Fried carrot cake', 'Ice kacang', 'Laksa', 'Mee hoon kueh',
          'Mee rebus', 'Mee siam', 'Nasi briyani', 'Nasi lemak', 'Oyster omelette', 'Popiah', 'Prawn noodles soup', 'Roti john', 'Roti prata', 'Satay', 'Wanton Mee']
fruit = ['Apple', 'Banana', 'Grapes', 'Kiwi', 'Mango', 'Orange', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

calories = get_calories()


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


@st.cache(suppress_st_warning=True)
def processed_img(img_path, model, class_names):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    print("answer is", answer)
    print("length of class_names", len(class_names))
    if int(len(class_names)) > 2:
        pred_class = class_names[tf.argmax(answer[0])]
        print("pred_class is", pred_class)
        pred_conf = tf.reduce_max(answer[0])
        print("pred_conf", pred_conf)
        top_5_i = sorted((answer.argsort())[0][-5:][::-1])
        print("top_5_i", top_5_i)
        values = answer[0][top_5_i] * 100
        print("values", values)
        labels = []
        for x in range(5):
            labels.append(class_names[top_5_i[x]])
            print("labels", labels)

        df = pd.DataFrame({"Top 5 Predictions": labels,
                           "F1 Scores": values,
                           'color': ['#d40b1f', '#720bd4', '#0b62d4', '#0bd4a5', '#0bd422']})
        df = df.sort_values('F1 Scores')
        print("df = ", df)
        st.success(f'Prediction : {pred_class} \n|| Confidence : {pred_conf * 100:.2f}%')
        st.write(alt.Chart(df).mark_bar().encode(
            x='F1 Scores',
            y=alt.X('Top 5 Predictions', sort=None),
            color=alt.Color("color", scale=None),
            text='F1 Scores'
        ).properties(width=600, height=400))
    else:
        pred_class = class_names[int(tf.round(answer))]
        print("pred_class is", pred_class)
        st.success(f'Prediction : {pred_class}')
    return pred_class.capitalize()


def run():
    st.image("https://github.com/DSstore/AIP/raw/main/snapfood.gif")
    app_mode = st.sidebar.selectbox("Choose the Classification Model",
                                    ["Binary Classification Model", "Multi-Class Classification Model",
                                     "EfficientNet Model_V1", "EfficientNet Model_V2"])

    if app_mode == "Binary Classification Model":
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Loss", "0.088")
        col2.metric("Accuracy", "96.5%")
        col3, col4 = st.sidebar.columns(2)
        col3.metric("Val_Loss", "0.35")
        col4.metric("Val_ Acc", "89.6%")

        model = load_model('Binary_CNN_model_6.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        class_names = get_2_classes()
        st.sidebar.write("Number of identified food:  ", len(class_names))
        st.sidebar.table(class_names)
    elif app_mode == "Multi-Class Classification Model":
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Loss", "0.19")
        col2.metric("Accuracy", "94%")
        col3, col4 = st.sidebar.columns(2)
        col3.metric("Val_Loss", "0.80")
        col4.metric("Val_ Acc", "81.48%")

        model = load_model('Multi_class_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        class_names = get_10_classes()
        st.sidebar.write("Number of identified food:  ", len(class_names))
        st.sidebar.table(class_names)
    elif app_mode == "EfficientNet Model_V1":
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Loss", "0.19")
        col2.metric("Accuracy", "98%")
        col3, col4 = st.sidebar.columns(2)
        col3.metric("Val_Loss", "0.43")
        col4.metric("Val_ Acc", "88.6%")

        model = load_model('efficientnet_model_1.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        class_names = get_20_classes()
        st.sidebar.write("Number of identified food:  ", len(class_names))
        st.sidebar.table(class_names)
    elif app_mode == "EfficientNet Model_V2":
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Loss", "0.0224")
        col2.metric("Accuracy", "100%")
        col3, col4 = st.sidebar.columns(2)
        col3.metric("Val_Loss", "0.4299")
        col4.metric("Val_ Acc", "86.86%")

        model = load_model('efficientnet_model_3_V2_35_classes')
        class_names = get_35_classes()
        st.sidebar.write("Number of identified food:  ", len(class_names))
        st.sidebar.table(class_names)

    st.sidebar.title("**Disclaimer**")
    st.sidebar.write(
        "Daily values are based on 2000 calorie diet and 155 lbs body weight.\nActual daily nutrient requirements might be different based on your age, gender, level of physical activity, medical history, and other factors.\nAll data displayed on this site is for general informational purposes only and should not be considered a substitute of a doctor's advice. Please consult with your doctor before making any changes to your diet.\nNutrition labels presented on this site is for illustration purposes only. Food images may show a similar or a related product and are not meant to be used for food identification.\nNutritional value of a cooked product is provided for the given weight of cooked food. \nData from USDA National Nutrient Database.")

    application_mode = st.radio(
        "Select your capture mode",
        ('Images', 'Camera'))
    if application_mode == 'Images':
        img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    else:
        img_file = st.camera_input("Take a picture")

    if not img_file:
        st.warning("Please upload an image")
        st.stop()
    else:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        pred_button = st.button("Predict")

    if pred_button:
        pred_class = processed_img(save_image_path, model, class_names)

        if pred_class in sgfood:
            st.info('**Category : Singapore Local Dish**')
            st.image(calories[pred_class])

        else:
            st.info('**Category : Fruits**')
            cal = fetch_calories(pred_class)
            if cal:
                st.warning('**' + cal + '(100 grams)**')


run()