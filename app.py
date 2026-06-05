import streamlit as st 
import numpy as np
from colorthief import ColorThief
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove
import pandas as pd
import io 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import  load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_data
def load_cnn() :
    model = load_model("models\cnn.keras")
    
    return model 


def predictions(model, imgs) :
    
    class_names = ['dress','hat', 'longsleeve','outwear','pants','shirt','shoes','shorts','skirt','t-shirt']
    
    pro_imgs = []
    for img in imgs :
        img = load_img(img,targe_size=(224,224))
        img_arr = img_to_array(img)
        
        pro_imgs.append(img_arr)
        
    raw_data = np.array(pro_imgs)
    ip = preprocess_input(raw_data)
    
    pred = model.predict(ip)
    res = tf.nn.softmax(pred, axis=1).numpy()
    pred_idx = np.argmax(res, axis=1)
    
    final_op =[]
    for i in range (len(pred_idx)):
        idx = pred_idx[i]
        label = class_names[idx]
        
        final_op.append({"Index":i, "Label" : label, "Image" : imgs[i]})
        
    df = pd.DataFrame(final_op)
    
    return df[df['Label' == "shirt"]],   df[df['Label' == "pants"]],  df[df['Label' == "shoes"]]
        


def val_to_df (imgs,label) :
    res1 = []
    res2 = []
    
    for idx, img in enumerate(imgs):
        res1.append([idx,img])
        
        raw_pixels = remove(img.read(), force_return_bytes=True)
        raw_data = Image.open(io.BytesIO(raw_pixels))
        rgb_data = Image.new("RGB", raw_data.size,(255,255,255))
        rgb_data.paste(raw_data,mask=raw_data.split()[3])
        img_arr = io.BytesIO()
        rgb_data.save(img_arr,format="jpeg")
        img_arr.seek(0)
        
        ct = ColorThief(img_arr)
        pal = ct.get_color(quality=1)
        res2.append(pal)
    df1 = pd.DataFrame(res1, columns=[f"{label}_Index", f"{label}_Image"])
    df2 = pd.DataFrame(res2, columns=[f"{label}_Red", f"{label}_Green", f"{label}_Blue"])
    df = pd.concat([df1,df2], axis=1 )
    
    return df


st.set_page_config("image shit", page_icon="📷", layout='wide')
imgs = st.file_uploader('Upload Images',type=['jepg','jpg','png'], accept_multiple_files=True)

model = load_cnn()

shirt_df,pants_df,shoes_df = predictions(model, imgs)

shirt_result = val_to_df(imgs)
pants_result = val_to_df(imgs)
shoes_result = val_to_df(imgs)

# st.dataframe(result.drop(columns=['Image']))


