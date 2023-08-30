import streamlit as st
import os
import pandas as pd
import joblib as jb

heading_style = '''
<div style="color:red;" align='center'>
<h1>Sales Prediction</h1>
</div>
'''
def return_df(tv,radio,newspaper):
    kbn={
   'tv':[tv],
   'radio':[radio],
   'newpaper':[newpaper]
    }   
    final_df=pd.DataFrame(kbn)
    return final_df


def base_model():
    bmodel=jb.load(os.path.join('finalized_model_regression.pkl'))
    return bmodel

st.markdown(heading_style, unsafe_allow_html=True)
tv=st.number_input('How much money was spent spent on tv advertiement?',min_value=0)
radio=st.number_input('How much money was spent spent on radio  advertiement?',min_value=0)
newpaper=st.number_input('How much money was spent spent on newpaper advertiement?',min_value=0)
df=return_df(tv,radio,newpaper)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	st.write(predictions)
