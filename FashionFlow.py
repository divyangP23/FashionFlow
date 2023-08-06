import streamlit as st
from PIL import Image
import tempfile
import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


st.set_page_config(
    page_title="FashionFlow",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="expanded",
)


def page1():
    st.header('FashionFlow ')

    similarity = pickle.load(open('similarity_matrix.pkl', 'rb'))

    fashion = pd.read_csv('processed_data.csv')

    categories = []
    for i in range(fashion['product_name'].shape[0]):
        categories.append(fashion['product_name'].iloc[i])

    # Define the options for the dropdown
    options = categories

    # Add the dropdown to the Streamlit app
    selected_option = st.selectbox('Select an apparel', categories)
    cv = CountVectorizer(max_features=3000,
                         stop_words='english')  # max_features means how many words you want to take out of say 50000 words
    vectors = cv.fit_transform(fashion['tags']).toarray()
    cv.get_feature_names_out()

    if st.button('Recommend'):
        # # Use the selected option
        st.write('Selected Apparel: ', selected_option)

        selected_row = fashion[fashion['product_name'] == selected_option]
        selected_img_link = selected_row['img_link'].values[0]

        # Display the image
        st.image(selected_img_link, caption='Selected Apparel')

        fashion_index = fashion[fashion['product_name'] == selected_option].index[
            0]  # this is extracting movie index and not movie id
        distances = list(enumerate(similarity[fashion_index]))
        fashion_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

        recommended_fashion = []
        for i in fashion_list:
            recommended_fashion.append(fashion.iloc[i[0]].product_name)

        st.write('Recommended Apparel')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            recommended_row = fashion[fashion['product_name'] == recommended_fashion[0]]
            recommended_img_link = recommended_row['img_link'].values[0]

            # Display the image
            st.image(recommended_img_link, caption=recommended_fashion[0])
        with col2:
            recommended_row = fashion[fashion['product_name'] == recommended_fashion[1]]
            recommended_img_link = recommended_row['img_link'].values[0]

            # Display the image
            st.image(recommended_img_link, caption=recommended_fashion[1])
        with col3:
            recommended_row = fashion[fashion['product_name'] == recommended_fashion[2]]
            recommended_img_link = recommended_row['img_link'].values[0]

            # Display the image
            st.image(recommended_img_link, caption=recommended_fashion[2])
        with col4:
            recommended_row = fashion[fashion['product_name'] == recommended_fashion[3]]
            recommended_img_link = recommended_row['img_link'].values[0]

            # Display the image
            st.image(recommended_img_link, caption=recommended_fashion[3])
        with col5:
            recommended_row = fashion[fashion['product_name'] == recommended_fashion[4]]
            recommended_img_link = recommended_row['img_link'].values[0]

            # Display the image
            st.image(recommended_img_link, caption=recommended_fashion[4])

def page2():
    st.header('FashionFlow ')

    uploaded_file = st.file_uploader("Upload the Apparel ", type=["jpg", "jpeg", "png"])

    feature = np.array(pickle.load(open('features.pkl', 'rb')))
    filenames = pickle.load(open('filename.pkl', 'rb'))
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        file_name = uploaded_file.name
        st.image(file_contents, caption="Uploaded Image")
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalised_result = result / norm(result)
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature)
        distances, indices = neighbors.kneighbors([normalised_result])
        st.write("Recommended Apparels:")
        # print(type(indices))
        rec = []
        for file in indices[0][1:6]:
            rec.append(filenames[file])
            # print(type(temp))
        # print(rec[0])
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            # print(rec[0])
            st.image(rec[0])
        with col2:
            st.image(rec[1])
        with col3:
            st.image(rec[2])
        with col4:
            st.image(rec[3])
        with col5:
            st.image(rec[4])

def main():
    selected_page = st.sidebar.radio("Categories", ("By using images","By using text prompt"))

    if selected_page == 'By using text prompt':
        page1()
    elif selected_page == 'By using images':
        page2()


if __name__ == "__main__":
    main()