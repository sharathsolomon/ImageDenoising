import streamlit as st
import cv2 
import numpy as np    
import tensorflow as tf
#import pandas as pd
import time
#from PIL import Image
import os
from patchify import patchify, unpatchify

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('About the App','Use the App')
        )
        
    #readme_text = st.markdown(get_file_content_as_string("README.md"))
    
    if selected_box == 'Use the App':
        #readme_text.empty()
        models()
        

def models():
    st.title("Image Denoising using Deep Learning")
    st.subheader('You can predict on sample images or you can upload a noisy image and get its denoised output.')
    
    col1,col2 = st.beta_columns(2)
    image = col1.file_uploader('Upload the image below')

    col1,col2 = st.beta_columns(2)
    predict_button = col1.button('Predict on uploaded image')
    sample_data = col2.button('Predict on sample images')
    
    if sample_data:
        option = st.sidebar.selectbox('Select a sample image',('Toy car','Vegetables','Gadget desk','Srabble board','Shoes','Door','A note'))
        #path = os.path.join(os.getcwd())#,'NOISY/')
       
        st.subheader('Noisy Image')
        nsy_img = cv2.imread(option+'.jpg')
        st.image(nsy_img)
        submit = st.button('Predict Now')
        if submit:
            prediction(nsy_img)
        
            
    elif predict_button:
        if image is not None:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            nsy_img = cv2.imdecode(file_bytes, 1)
            #nsy_img = cv2.imread(image)
            #st.image(nsy_img,channels='RGB')
            #nsy_img = cv2.cvtColor(nsy_img, cv2.COLOR_BGR2RGB)
            #st.subheader('Noisy Image')
            #st.image(nsy_img)
            prediction(nsy_img)
        else:
            st.text('Please upload the image')    
            
def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

@st.cache
def get_model():
    RIDNet=tf.keras.models.load_model('RIDNet.h5')
    return RIDNet

def prediction(img):
    model = get_model()
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(1024,1024))
    img = img.astype("float32") / 255.0

    img_patches = patches(img,256)
    
    nsy=[]
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)
    
    pred_img = model.predict(nsy)
    pred_img = np.reshape(pred_img,(4,4,1,256,256,3))
    pred_img = unpatchify(pred_img, img.shape)
    end = time.time()
    
    #col1,col2 = st.beta_columns(2)
    #with col1:
    #    st.header("Noisy Image")
    #    st.image(img)
    #with col2:
    #    st.header("Predicted Image")
    #    st.image(pred_img)    
    img = cv2.resize(img,(256,256))
    pred_img = cv2.resize(pred_img,(256,256))
    st.subheader("Noisy Image")
    st.image(img) 
    st.subheader("Predicted Image")
    st.image(pred_img)   
    st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    
if __name__ == "__main__":
    main()    
        
        
        
        
        
        
        
    
