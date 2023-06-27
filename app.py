import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


st.set_page_config(layout='wide')
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )
add_bg_from_local('Images/bg.jpg')

df=pd.read_csv(r'E:\DATASETS\creditcard.csv')

head1,head2,head3=st.columns([0.65,1,0.5])
st.write('#')
with head2:
    st.title('Creditcard Fraud Detection')
col1,col2,col3=st.columns([1,0.5,1])

with col1:
    v1=st.number_input('V1')
    v2=st.number_input('V2')
    v3=st.number_input('V3')
    v4=st.number_input('V4')
    v5=st.number_input('V5')
    v6=st.number_input('V6')
    v7=st.number_input('V7')
    v8=st.number_input('V8')
    v9=st.number_input('V9')
    v10=st.number_input('V10')
    v11=st.number_input('V11')
    v12=st.number_input('V12')
    v13=st.number_input('V13')
    v14=st.number_input('V14')
    v15=st.number_input('V15')
with col3:
    v16=st.number_input('V16')
    v17=st.number_input('V17')
    v18=st.number_input('V18')
    v19=st.number_input('V19')
    v20=st.number_input('V20')
    v21=st.number_input('V21')
    v22=st.number_input('V22')
    v23=st.number_input('V23')
    v24=st.number_input('V24')
    v25=st.number_input('V25')
    v26=st.number_input('V26')
    v27=st.number_input('V27')
    v28=st.number_input('V28')
    amount=st.number_input('Amount')

btncol1,btncol2,btncol3=st.columns([2.3,0.5,2])
with btncol2:
    st.write('#')
    predict=st.button('Predict')

if predict==True:
    scale=pickle.load(open('scaler.pkl','rb'))
    model=pickle.load(open('model.pkl','rb'))
    scaled_amount=scale.transform(np.array([amount]).reshape(-1,1))
    values=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,scaled_amount]
    result=model.predict(np.array(values).reshape(1,-1))
    print(result)
    res1,res2,res3=st.columns([2.3,1,2])
    with res2:
        if result[0]==1:
            def add_bg_from_local(image_file):
                with open(image_file, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                st.markdown(
                    f"""
                <style>
                .stApp {{
                    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                    background-size: cover
                }}
                </style>
                """,
                    unsafe_allow_html=True
                )
            add_bg_from_local('Images/fraud.jpg')
            # st.markdown(
            #     "<h2 style= 'color: #7B1FA2;font-size: 60px;'><b>Fraud</b></h2>",
            #     unsafe_allow_html=True)
            st.image('Images/fraudalert.png')

        else:
            def add_bg_from_local(image_file):
                with open(image_file, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                st.markdown(
                    f"""
                <style>
                .stApp {{
                    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                    background-size: cover
                }}
                </style>
                """,
                    unsafe_allow_html=True
                )
            add_bg_from_local('Images/notfraud.jpg')
            # st.markdown(
            #     "<h2 style= 'color: #64FFF6;font-size: 30px;'>Not Spam</h2>",
            #     unsafe_allow_html=True)
            st.image('Images/notfraudtran.jpg')