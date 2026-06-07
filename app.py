import streamlit as st 
import numpy as np
from colorthief import ColorThief
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove
import pandas as pd
import io 
import os 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import  load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
from gradio_client import Client, handle_file
from huggingface_hub import InferenceClient
from io import BytesIO
import requests
import base64



@st.cache_resource
def load_tools() :
    cnn = load_model(r"models\cnn.keras")
    ann = load_model(r"models\ann.keras")
    scale = joblib.load(r"models\scale.plk")
    return cnn, ann , scale



@st.cache_data
def process_single_img(img):
    raw_pixels = remove(img.read(), force_return_bytes=True)
    raw_data = Image.open(io.BytesIO(raw_pixels))
    rgb_data = Image.new("RGB", raw_data.size,(255,255,255))
    rgb_data.paste(raw_data,mask=raw_data.split()[3])
    img_arr = io.BytesIO()
    rgb_data.save(img_arr,format="jpeg")
    img_arr.seek(0)
    
    ct = ColorThief(img_arr)
    pal = ct.get_color(quality=1)
    
    return pal




def clean_and_trim(raw_img):
    """Removes the background using rembg and crops tightly around the garment."""
    no_bg = remove(raw_img).convert("RGBA")
    bbox = no_bg.getbbox()
    if bbox:
        return no_bg.crop(bbox)
    return no_bg

@st.cache_data
def rendered_img(imgs, val):
    
    
    # 1. Clean and trim all 3 assets locally
    shirt = clean_and_trim(Image.open(imgs[0]))
    pants = clean_and_trim(Image.open(imgs[1]))
    shoes = clean_and_trim(Image.open(imgs[2]))
    
    shirt_w = 500
    pants_w = 420
    shoes_w = 360
    
    shirt_h = int(shirt_w * (shirt.height / shirt.width))
    shirt_res = shirt.resize((shirt_w, shirt_h))
    
    pants_h = int(pants_w * (pants.height / pants.width))
    pants_res = pants.resize((pants_w, pants_h))
    
    shoes_h = int(shoes_w * (shoes.height / shoes.width))
    shoes_res = shoes.resize((shoes_w, shoes_h))
    
    gap = 20
    canvas_w = max(shirt_w, pants_w, shoes_w) + 100 
    total_h = shirt_h + pants_h + shoes_h + (gap * 2) + 60
    
    outfit_canvas = Image.new("RGBA", (canvas_w, total_h), (0, 0, 0, 0))
    
    shirt_x = (canvas_w - shirt_w) // 2
    shirt_y = 30
    outfit_canvas.paste(shirt_res, (shirt_x, shirt_y), shirt_res)
    
    pants_x = (canvas_w - pants_w) // 2
    pants_y = shirt_y + shirt_h + gap
    outfit_canvas.paste(pants_res, (pants_x, pants_y), pants_res)
    
    shoes_x = (canvas_w - shoes_w) // 2
    shoes_y = pants_y + pants_h + gap
    outfit_canvas.paste(shoes_res, (shoes_x, shoes_y), shoes_res)
    
    final_output_path = f"clean_outfit_combination_{val}.png"
    outfit_canvas.resize((300,700), Image.Resampling.LANCZOS)
    outfit_canvas.save(final_output_path, format="PNG",)
    
    return final_output_path
    
    
    
def cat_predictions(model, imgs) :
    
    # class_names = ['shirt','pants', 'shoes']
    class_names = ['dress','hat', 'longsleeves','outweare','pants','shirt','shoes','shorts','skirt','t-shirt']
    shirt = ['dress','t-shirt','longsleeves','outweare','shirt']
    pants = ['shorts','skirt','pants']
    
    pro_imgs = []
    for img in imgs :
        img = load_img(img,target_size=(224,224))
        img_arr = img_to_array(img)
        
        pro_imgs.append(img_arr)
        
    raw_data = np.array(pro_imgs)
    # ip = preprocess_input(raw_data)
    
    pred = model.predict(raw_data)
    res = tf.nn.softmax(pred, axis=-1).numpy()
    pred_idx = np.argmax(res, axis=1)
    
    final_op =[]
    for i in range (len(pred_idx)):
        idx = pred_idx[i]
        label = class_names[idx]
        
        final_op.append({"Index":i, "Label" : label, "Image" : imgs[i]})
        
    df = pd.DataFrame(final_op)
    # st.dataframe(df.drop(columns='Image'))
    return df[df['Label'].isin(shirt)],   df[df['Label'].isin(pants)],  df[df['Label'].isin(["shoes"])]
        



def val_to_df (imgs,label) :
    res1 = []
    res2 = []
    
    for idx, img in enumerate(imgs):
        res1.append([idx,img])
        
        if hasattr(img, 'seek'):
            img.seek(0)
        
        pal = process_single_img(img)
        res2.append(pal)
    df1 = pd.DataFrame(res1, columns=[f"{label}_Index", f"{label}_Image"])
    df2 = pd.DataFrame(res2, columns=[f"{label}_Red", f"{label}_Green", f"{label}_Blue"])
    df = pd.concat([df1,df2], axis=1 )
    
    return df



def combo_predictions(model, scale, df) :
    unwanted = ['shirt_Image','pants_Image','shoes_Image', 'shirt_Index','pants_Index','shoes_Index']
    scaled_df = scale.fit_transform(df.drop(columns=unwanted))
    
    pred = model.predict(scaled_df)
    pred = pd.DataFrame(pred, columns=[ "Rating"]).reset_index()
    fin_df = pd.concat([df,pred], axis=1)
    
    return fin_df



st.set_page_config("COMBO FINDER 👕👖", page_icon="📷", layout='wide')

st.title('Find the best combo',text_alignment='center')

st.markdown(
    """
    <style>
    /* 1. Main app gradient background */
    .stApp {
        background: linear-gradient(
            135deg, 
            #0a0a0c 0%,    /* Deep Jet Black */
            #0b1b3d 25%,   /* Midnight Blue */
            #0d3a2f 45%,   /* Deep Emerald/Teal */
            #5c134f 70%,   /* Rich Violet */
            #d62196 100%   /* Electric Fuchsia/Pink */
        );
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* 2. Custom Spinner Styling */
    /* This overrides the loading wheel to match the pink/violet theme */
    div[data-testid="stSpinner"] > div {
        border-top-color: #d62196 !important;    /* Spinning edge (Electric Fuchsia) */
        border-right-color: #5c134f !important;  /* Secondary spinning edge (Rich Violet) */
        border-bottom-color: rgba(255, 255, 255, 0.1) !important; /* Faded track */
        border-left-color: rgba(255, 255, 255, 0.1) !important;   /* Faded track */
    }
    
    /* Stylized container for the spinner text */
    div[data-testid="stSpinner"] p {
        color: #F0E6D2 !important; /* Elegant champagne color for loading text */
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 1px;
        font-weight: 400;
    }

    /* 3. Typography typography */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Cinzel', 'Playfair Display', serif;
        letter-spacing: 3px;
        text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    
    p, span, label {
        color: #E2E8F0 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.columns(3)[0]:
    st.subheader("Upload Images Here")

imgs = st.file_uploader(' ',type=['jepg','jpg','png'], accept_multiple_files=True)



btn = st.button('Predict')
cnn, ann, scale = load_tools()
if btn and imgs :
    shirt_df,pants_df,shoes_df = cat_predictions(cnn, imgs)

    

    shirt_result = val_to_df(shirt_df["Image"].tolist(),"shirt")
    pants_result = val_to_df(pants_df["Image"].tolist(),"pants")
    shoes_result = val_to_df(shoes_df["Image"].tolist(),"shoes")
    
    ann_df = pd.merge(pd.merge(shirt_result,pants_result, how='cross'),shoes_result, how= 'cross')

    

    output = combo_predictions(ann, scale, ann_df).sort_values('Rating', ascending=False).reset_index(drop=True)
    # st.dataframe(output.drop(columns=["shirt_Image",'pants_Image','shoes_Image']))
    output = output.drop_duplicates(subset=['shirt_Index'], keep="first").reset_index(drop=True)
    # output = output.drop_duplicates(subset=['pants_Index'], keep="first").reset_index(drop=True)
    
    # st.dataframe(output.drop(columns=["shirt_Image",'pants_Image','shoes_Image']))
    
    
    combo1 = rendered_img(output.loc[0,["shirt_Image",'pants_Image','shoes_Image']].tolist(),1)
    combo2 = rendered_img(output.loc[1,["shirt_Image",'pants_Image','shoes_Image']].tolist(),2)
    combo3 = rendered_img(output.loc[2,["shirt_Image",'pants_Image','shoes_Image']].tolist(),3)
    combo4 = rendered_img(output.loc[3,["shirt_Image",'pants_Image','shoes_Image']].tolist(),4)
    combo5 = rendered_img(output.loc[4,["shirt_Image",'pants_Image','shoes_Image']].tolist(),5)
    combo6 = rendered_img(output.loc[5,["shirt_Image",'pants_Image','shoes_Image']].tolist(),6)
    
    if combo1 and combo2 and combo3 : 
        
        st.subheader("The Following are The Top 6 Combinations of your Wardrobe:", text_alignment='left')
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col1:
            with st.container(border=True,height= 500):
                st.image(combo1,caption=f"{round(float(output.loc[0,"Rating"]),1)} ⭐")
                
        with col2:
            with st.container(border=True,height= 500):
                st.image(combo2,caption=f"{round(float(output.loc[1,"Rating"]),1)} ⭐")
        with col3:
            with st.container(border=True,height= 500):
                st.image(combo3,caption=f"{round(float(output.loc[2,"Rating"]),1)} ⭐")
        
        with col4:
            with st.container(border=True,height= 500):
                st.image(combo4,caption=f"{round(float(output.loc[3,"Rating"]),1)} ⭐")
        
        with col5:
            with st.container(border=True,height= 500):
                st.image(combo5,caption=f"{round(float(output.loc[4,"Rating"]),1)} ⭐")
        
        with col6:
            with st.container(border=True,height= 500):
                st.image(combo6,caption=f"{round(float(output.loc[5,"Rating"]),1)} ⭐")
        