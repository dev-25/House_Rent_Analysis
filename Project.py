import numpy as pd
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.linear_model import  Ridge
import pymongo
from pymongo import MongoClient as mc

cluster = mc('mongodb+srv://dev:dev@project.isr0wio.mongodb.net/?retryWrites=true&w=majority')
db = cluster["DB"]
collection = db["Prediction"]

df = pd.read_csv("House_Rent_Dataset.csv")


st.set_page_config (page_title="House Rent Analysis", page_icon="",layout="wide")
st.write('<style>div.block-container{padding-left:1.4rem;padding-top:2rem;}</style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;background-color:#1f1833;border-radius:19px;'>🏡House Rent Analysis !!!</h1>",unsafe_allow_html=True)	

for i in range(3):
	st.write('')

coll1, coll2, coll3 = st.columns((1,2,1))
with coll2:
	option = st.selectbox('Select...',('Dataset Preview', 'Data Visualization', 'Prediction','DataBase'))
for i in range(3):
	st.write('')

if option =="Dataset Preview":
	st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>",unsafe_allow_html=True)	

	col1,padding, col2, col3 = st.columns((1,1.7,19,1))
	with col2:
			st.dataframe(df)

if option =='Data Visualization':
	for i in range(3):
		st.write('')
	st.markdown("<h2 style='text-align: center;'>Data Visualization</h2>",unsafe_allow_html=True)	

	padding,b1,padding = st.columns((2.8,6,3))
	with b1:
		features=['City']
		for feature in features:
		    fig3=px.bar(df[feature].value_counts(),width=800,color=df[feature].value_counts(),text=df[feature].value_counts(),title='Total Cities Present in Dataset')
		    fig3.update_layout(xaxis_title=feature,yaxis_title='No.of Houses')
		    st.plotly_chart(fig3)	


	padding,a1,padding,a2 = st.columns((0.3,2,0.3,2))
	with a1:
		fig=px.bar(df['BHK'].value_counts(),width=600,
	       color=df['BHK'].value_counts(),
	       text=df['BHK'].value_counts(),
	       title='BAR PLOT OF BED ROOMS',
	       template='plotly_dark')
		fig.update_layout(xaxis_title='Bed rooms count',yaxis_title='Total No.of Bedrooms')
		st.plotly_chart(fig)

		features=['Area Type']
		for feature in features:
		    fig1=px.bar(df[feature].value_counts(),width=600,color=df[feature].value_counts(),text=df[feature].value_counts(),title=feature)
		    fig1.update_layout(xaxis_title=feature,yaxis_title='No.of Houses')
		    st.plotly_chart(fig1)


	with a2:
		features=['Furnishing Status','Point of Contact']
		for feature in features:
		    fig2=px.bar(df[feature].value_counts(),width=600,color=df[feature].value_counts(),text=df[feature].value_counts(),title=feature)
		    fig2.update_layout(xaxis_title=feature,yaxis_title='No.of Houses')
		    st.plotly_chart(fig2)

	for i in range(5):
		st.write('')
	st.markdown("<h3 style='text-align: center;'>City Rents</h3>",unsafe_allow_html=True)	

	padding,c1,padding,c2 = st.columns((0.3,2,0.3,2))
	with c2:
		fig4 = px.scatter(x='Rent',y='Size',data_frame=df,color="Size",title="Rent Based on House Size",width=600)
		st.plotly_chart(fig4)

		fig5 = px.scatter(df,x='BHK',y='Rent',title="Rent Based on BHK",color="BHK",width=600)
		st.plotly_chart(fig5)

	with c1:
		fig6 = px.scatter(df,x='Rent',y='City',title="Rent Based on City",color="City",width=600)
		st.plotly_chart(fig6)

		fig7 = px.scatter(df,x='Rent',y='Area Type',title="Rent Based on Area Type",color="Area Type",width=600)
		st.plotly_chart(fig7)

target = "Rent"
featuree = ["City","BHK","Size"]
y=df[target]
X=df[featuree]
cutoff = int(len(X) * 0.8)

X_train, y_train = X.iloc[:cutoff],y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:],y.iloc[cutoff:]

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)
model.fit(X_train,y_train)

def make_prediction(data):
    df = pd.DataFrame(data,index=[0])
    prediction = model.predict(df).round(2)[0]
    # return f"Predicted apartment rent: {prediction}"
    return prediction    


if option=="Prediction":
	aaa,sss,fff = st.columns((1,2,1))
	with sss:
		st.markdown("<h2 style='text-align: center;'>House Rent Prediction</h2>",unsafe_allow_html=True)
		option1 = st.selectbox('Select a city',('Mumbai','Chennai','Bangalore',"Hyderabad",'Delhi','Kolkata'))
		bhkk = st.number_input('Enter BHK', 1,6)
		sizee = st.number_input('Enter Size for House', 10,8000)
		# bth = st.number_input('How much Washromm', 1,10)	
		data = {"City":option1,"BHK":bhkk,"Size":sizee}

		if st.button('Predict'):
			a = make_prediction(data)
			st.subheader(f"Predicted apartment rent: ₹{a}")

			data1 = {"City":option1,"BHK":bhkk,"Size":sizee, "Predicted Rent":a}
			collection.insert_one(data1)

if option=="DataBase":
	st.write('')
	aa,d1,d2,d3,d4,d5,asd = st.columns((2,2,2,2,2,1,2))

	with d1:
		st.markdown("<h3 style='color:#66fcf1;text-align: center;'>City</h3>",unsafe_allow_html=True)
		st.write('')
		for record in collection.find({},{ "_id": 0,"City":1 }):
			for v in record.values():              
				def header(v):                 
					st.markdown(f'<p style="border-style: solid;border-color:#66fcf1;background-color:#1f1833;text-align: center;padding:18.2px; color:white; font-size:16px; border-radius:8px;">{v}</p>', unsafe_allow_html=True)
				header(v)		

	with d2:
		st.markdown("<h3 style='color:#66fcf1;text-align: center;'>BHK</h3>",unsafe_allow_html=True)
		st.write('')
		for record in collection.find({},{ "_id": 0,"BHK":1 }):
			for v1 in record.values():              
				def header(v1):                 
					st.markdown(f'<p style="border-style: solid;border-color:#66fcf1;background-color:#1f1833;text-align: center;padding:18.2px; color:white; font-size:16px; border-radius:8px;">{v1}</p>', unsafe_allow_html=True)
				header(v1)		

	with d3:
		st.markdown("<h3 style='color:#66fcf1;text-align: center;'>Size</h3>",unsafe_allow_html=True)
		st.write('')
		for record in collection.find({},{ "_id": 0,"Size":1 }):
			for v2 in record.values():              
				def header(v2):                 
					st.markdown(f'<p style="border-style: solid;border-color:#66fcf1;background-color:#1f1833;text-align: center;padding:18.2px; color:white; font-size:16px; border-radius:8px;">{v2}</p>', unsafe_allow_html=True)
				header(v2)		

	with d4:
		st.markdown("<h3 style='color:#66fcf1;text-align: center;'>Predicted Rent</h3>",unsafe_allow_html=True)
		st.write('')
		for record in collection.find({},{ "_id": 0,"Predicted Rent":1 }):
			for v3 in record.values():              
				def header(v3):                 
					st.markdown(f'<p style="border-style: solid;border-color:#66fcf1;background-color:#1f1833;text-align: center;padding:18.2px; color:white; font-size:16px; border-radius:8px;">{v3}</p>', unsafe_allow_html=True)
				header(v3)		

	with d5:
		for i in range(1,6):
			st.write('')
		for record in collection.find({},{ "_id": 1}):
			for vv in record.values():
				if st.button("🗑",key=vv):
					delete = {"_id": vv}
					collection.delete_one(delete)
					st.experimental_rerun()			
				st.markdown("""<style>.stButton > button {border-style: solid;border-color:#66fcf1;color: white;margin:-4.6px;background: #1f1833;width: 50px;height: 60px;}</style>""", unsafe_allow_html=True)