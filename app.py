import os
import base64
import pickle
import numpy as np
import pandas as pd
import smtplib
from deta import Deta
import streamlit as st
import tensorflow as tf
from fonction import *
from dotenv import load_dotenv
from keras.models import load_model
import streamlit.components.v1 as stc
import streamlit_authenticator as stauth
from dependancies import sign_up, fetch_users, send_email_tab, send_email_img

#Load the environment variables
load_dotenv(".env")
#DETA_KEY = os.getenv("DETA_KEY")
DETA_KEY = 'a0m78dwvhdm_1SDma2An8odFQf6Ti6QN2Nr9FxjztVbi'

deta = Deta(DETA_KEY)

db = deta.Base('FedIA')

# Define the HTML code for Titre
HTML_BANNER = """
    <div style="background-color:#464e5f;top: 0;padding:10px;border-radius:10px;">
        <h1 style="color:white;text-align:center;">FedAI Stroke Prediction</h1>
        <p style="color:white;text-align:center;">Application</p>
    </div>
    """

st.set_page_config(page_title="FedIA", page_icon="", layout="centered")

# User Authentification
users = db.fetch().items
#print(users)
#emails = []
roles = []
names = []
usernames = []
passwords = []

for user in users:
    #emails.append(user['key'])
    names.append(user['name'])
    roles.append(user['role'])
    usernames.append(user['username'])
    passwords.append(user['password'])

credentials = {'usernames': {}}
for index in range(len(names)):
    credentials['usernames'][usernames[index]] = {'name': names[index], 'password': passwords[index]}

authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)

name, authentication_status, username = authenticator.login(':green[Login]', 'main')

    
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

#st.session_state


def show_logout_page():
    with logOutSection:
        # Message de bienvenue
        st.sidebar.title(f"Welcome {name}")
        #logout, add = st.columns(2)
        #with logout:
        authenticator.logout("Log Out", "sidebar")
        #with add:
        #    adduser = st.sidebar.button("Add user")
        #    if adduser:
        #        sign_up()
      
def show_login_page():
    with loginSection:
        info, info1 = st.columns(2)

        if username:
            if username in usernames:
                if authentication_status:
                    #if st.session_state.
                    loginSection.empty()
                    # let User see app
                    show_logout_page()
                    show_main_page()
                elif not authentication_status:
                    with info:
                        st.error('Incorrect Password or username')
                else:
                    with info:
                        st.warning('Please feed in your credentials')
            else:
                with info:
                   st.warning('Username does not exist, Please Sign up')
    
            
@st.cache_resource
def dataset():
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    df = df.drop('id', axis=1)
    return df
    #st.dataframe(dataframe)   

def show_main_page():
    with mainSection:        
        # Section de la barre latérale
        with st.sidebar:
            selected = option_menu("Stroke Prediction", ["Home", 'Support'], icons=['house', 'gear'], menu_icon="cast", default_index=0, orientation="horizontal")
            selected

        # Section Principal
        # Page de rapports
        if selected == "Home":
            # Tite page
            stc.html(HTML_BANNER)
            #st.title("FedAI Stroke Prediction")
            st.markdown(
                    """
                    ---
                    """
                )
            #Image d'accueil
            image = Image.open('/Users/harouna/Documents/PFE/Stroke/St.jpg')

            st.image(image) #caption="Resized Image", use_column_width=True
            st.markdown(
                    """
                    ---
                    """
                )
            uploaded_file = st.file_uploader("Choose file to view the report")
            if uploaded_file is not None:
                #dataset()
                df = pd.read_csv(uploaded_file)
                df = df.drop('id', axis=1)
                
                num_col = df.select_dtypes(exclude=['object']).columns.to_list()
                cat_col = df.select_dtypes(include=['object']).columns.to_list()
                
                # Displaying properties of dataset
                tab1, tab2 = st.tabs(["Distribution by characteristic", "Histogram", ])
                with tab1:
                   st.header("Distribution by characteristic")
                   countplot(df,num_col,cat_col)

                with tab2:
                    #Tite page
                    st.header("Histogram")
                    histogramme_to_choice(df,num_col,cat_col)
          
        # Tabulaire Prediction page
        else:
            st.title("Prediction")
            tab1, tab2, tab3 = st.tabs([ 
                "Prediction with tabulaires datas", 
                "Prediction with images datas",
                "Prediction with patient"
                ])
            
            with tab1:
                #Tite page
                st.header("Enter the parameters to predict")
                
                #Load model
                modeltab = load_model('./Models/Model_tab.h5')
                
                col1, col2 = st.columns(2)
                # Store the initial value of widgets in session state
                if "visibility" not in st.session_state:
                    st.session_state.visibility = "visible"
                    st.session_state.disabled = False

                with col1:
                    gender = st.selectbox("Gender", ("Male","Female","Other"),
                    label_visibility=st.session_state.visibility,
                    disabled=st.session_state.disabled,)

                with col2:
                    age = st.text_input("Age")

                with col1:
                    work_type= st.selectbox("Work Type",("Private", "Govt_job", "Never_worked", "Self-employed", "children"))

                with col2:
                    hypertension = st.text_input("Hypertension")

                with col1:
                    ever_married= st.selectbox("Ever Married",("No","Yes"))

                with col2:
                    heart_disease = st.text_input("Heart Disease")

                with col1:
                    Residence_type= st.selectbox("Residence Type", ("Rural", "Urban"))

                with col2:
                    avg_glucose_level= st.text_input("Avg Glucose Level")

                with col1:
                    smoking_status= st.selectbox("Smoking Status",("formerly smoked", "never smoked", "smokes", "Unknown"))

                with col2:
                    bmi= st.text_input("BMI")
  
                feature = [[
                            gender,
                            age,
                            hypertension,
                            heart_disease,
                            ever_married,
                            work_type,
                            Residence_type,
                            avg_glucose_level,
                            bmi,
                            smoking_status
                        ]]
                #print(type(feature))
                predButton = st.button("Predict", disabled=False)

                #
                if predButton:
                    prediction = predict_tabulaire_data(modeltab,feature)
                    #stc.html(HTML_DIALOG.format(result=prediction) + JS_DIALOG)
                    #print(prediction)
                    df_pred = pd.DataFrame(feature,
                                           columns = ["Gender","Age","Hypertension","Heart_disease","Ever_married","Work_type","Residence_type","Avg_glucose_level","Bmi","Smoking_status"]
                                           )
                    df_pred['Stroke'] = ["Yes" if prediction >0.5 else "No"]
                    
                    #export_dataframe_to_csv(df_pred, csv_file_path)
                    df_pred.to_csv('./file.csv', index=False)
                    csv_file_path = './file.csv'
                    
                    if prediction > 0.50 :
                        #stc.html(HTML_DIALOG.format(result=pred) + JS_DIALOG)
                        st.error("This patient's risk of stroke is : {:.2%}.".format(prediction), icon="⚠️")
                        #Send email
                        user_email = "asdjim.espoir.guelmian@horizon-tech.tn"
                        send_email(user_email, name, prediction, csv_file_path)
                        
                    else:
                        prediction = 1-prediction
                        st.success("The patient is not at risk of stroke : {:.3%}".format(prediction), icon="✅")
                
                    st.dataframe(df_pred, hide_index=True)
                    
                    with open(csv_file_path, 'r') as f:
                        csv = f.read()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="file.csv">Download the file in CSV format</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with tab2:
                st.header("Select image to predict")
                
                # upload file
                file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])
                
                # load classifier
                modelImg = load_model('./Models/Model_image.h5')

                # display image
                if file is not None:
                    image = Image.open(file)
                    image = np.array(image)
                    size = (300, 300)
                    images = cv2.resize(image, size , interpolation=cv2.INTER_AREA)
                    st.image(images)

                    pred = imagePred(image, modelImg)
                    if pred > 0.50 :
                        st.error("This patient's risk of stroke is : {:.3%}".format(pred), icon="⚠️")
                        user_email = "asdjim.espoir.guelmian@horizon-tech.tn"
                        send_email_img(user_email, name, pred)
                    else:
                        pred = 1-pred
                        st.success("The patient is not at risk of stroke : {:.3%}".format(pred), icon="✅")
                
            with tab3:
                st.header("Select all images of patient to predict")
                #slices_path = st.text_input("Enter path to slices:", "path/to/slices")
                model = load_model("./Models/Stroke-prediction-123-0.8630.h5")
                # upload file
                #all_img = None
                
                #if all_img is None or len(all_img) == 0:
                all_img = st.file_uploader("", type=['jpeg', 'jpg', 'png'], accept_multiple_files=True)

                if all_img is not None and len(all_img) > 0:
                    scan = np.dstack(tuple(remove_noise(imageio.imread(img)) for img in all_img))
                    scan = normalize_scan(scan)
                    scan = resize_scan(scan)
                    
                    #Afficher le scan du patient
                    plot_scan_from_dataset(4, 16, 128, 128, scan, "Patient scan")
                    
                    prediction = predict_patient(scan, model)
                    if prediction > 0.50 :
                        st.error("This patient's risk of stroke is : {:.3%}".format(prediction), icon="⚠️")
                        user_email = "asdjim.espoir.guelmian@horizon-tech.tn"
                        send_email_img(user_email, name, prediction)
                    else:
                        pred = 1-prediction
                        st.success("The patient is not at risk of stroke : {:.3%}".format(pred), icon="✅")
                all_img = None

                
with headerSection:
    if "authentication_status" not in st.session_state:
        show_login_page()
    else:
        if st.session_state["authentication_status"]:
            show_logout_page()
            show_main_page()
        else:
            show_login_page()
