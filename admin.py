import os
import json
import pandas as pd
from db_fxns import *
import streamlit as st
from deta import Deta
from dependancies import *
import plotly.express as px
#from dotenv import load_dotenv
import streamlit.components.v1 as stc
from dependancies import sign_up, fetch_users
from streamlit_option_menu import option_menu


#Load the environment variables
#load_dotenv(".env")
#DETA_KEY = os.getenv("DETA_KEY")

DETA_KEY = 'a0m78dwvhdm_1SDma2An8odFQf6Ti6QN2Nr9FxjztVbi'
deta = Deta(DETA_KEY)

db = deta.Base('FedIA')


HTML_USER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Gestion des Utilisateurs</h1>
    <!--<p style="color:white;text-align:center;">Built with Streamlit</p>-->
    </div>
    """

# User Authentification
users = db.fetch().items#fetch_users()
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

print(credentials)
st.session_state

def gestionUsers():
    stc.html(HTML_USER)
    
    with st.sidebar:
        selected = option_menu("User Admin", ["List", 'Add', 'Update', 'Delete'], menu_icon="cast", default_index=0, orientation="vertical")
        selected

    # Section Principal
    # Page de rapports
    if selected == "List":
        # Tite page
        st.subheader("List Users")
        #st.markdown(
        #        """
        #        ---
        #        """
        #    )
        with st.expander("All Users"):
            listUsers = fetch_users()
            listUsers = pd.DataFrame(listUsers)
            st.dataframe(listUsers)
        
        with  st.expander("Role Status"):
            listRole = listUsers["role"].value_counts().to_frame()
            listRole = listRole.reset_index()
            st.dataframe(listRole)
            
            p1 = px.pie(listRole,values='count',names='role')
            st.plotly_chart(p1, use_container_width=True)

    elif selected =="Add":
        st.subheader("Add User")
        sign_up()
    
    elif selected == ("Update"):
        st.subheader("Update User")
        
        with st.expander("Current Users"):
            listUsers = fetch_users()
            listUsers = pd.DataFrame(listUsers)
            st.dataframe(listUsers)
            
        res = fetch_users()
        name = [user['key'] for user in res]
        updated_User = st.selectbox("Select user", name)
        mod = db.get(updated_User)
        #st.write(mod)
        
        if mod:
            key = mod['key']
            name = mod['name']
            role = mod['role']
            username = mod['username']
            passw = mod['password']
            #st.write(username)
            
            with st.form(key='Update', clear_on_submit=True):
                st.subheader(':green[Update User]')
                new_email = st.text_input(':blue[Email]', key)
                new_name = st.text_input(':blue[Name]', name)
                new_username = st.text_input(':blue[Username]', username)
                new_role = st.selectbox(':blue[Role]',("User", "Administrateur"), placeholder='Choice Role')
                new_password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
                new_password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')
                
                if st.form_submit_button('Update'):                
                #    hashed_password = stauth.Hasher([
                    key = "espoi@gmail.com"
                    db.update(mod,key)

                #if st.form_submit_button('Update'):                
                #    hashed_password = stauth.Hasher([new_password2]).generate()
                #    new_password = hashed_password[0]
                #    update_user(new_email, new_name, new_username, new_password, new_role)
                #if email and name and username:
                """if new_email:
                    if validate_email(new_email):
                        if validate_username(new_username):
                            if len(new_username) >= 2:
                                if len(new_password1) >= 6:
                                    if new_password1 == new_password2:
                                        # Add User to DB
                                        hashed_password = stauth.Hasher([new_password2]).generate()
                                        new_password = hashed_password[0]
                                        update_user(new_email, new_name, new_username, new_password, new_role)
                                        st.success('User information updated successfully!!')
                                        st.balloons()
                                    else:
                                        st.warning('Passwords Do Not Match')
                                else:
                                    st.warning('Password is too Short')
                            else:
                                st.warning('Username Too short')
                        else:
                            st.warning('Invalid Username')
                    else:
                        st.warning('Invalid Email')
                #else:
                #    st.warning('Please fill in all required fields')
                btn1, bt2, btn3, btn4, btn5 = st.columns(5)

                with btn3:
                    st.form_submit_button('Update')"""
        
    else:
        st.subheader("Delete User")
        
        res = fetch_users()
        email = [user['key'] for user in res]
        deleted_User = st.selectbox("Select user", email)
        
        if st.button("Delete"):
            db.delete(deleted_User)
            st.info("Deleted : '{}'".format(deleted_User))
        
        with st.expander("Current Users"):
            listUsers = fetch_users()
            listUsers = pd.DataFrame(listUsers)
            st.dataframe(listUsers)
            
def show_logout_page():
    #with logOutSection:
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
    #with loginSection:
    info, info1 = st.columns(2)
    if username:
        if username in usernames:
            if authentication_status:
                #if st.session_state.
                loginSection.empty()
                # let User see app
                show_logout_page()
                #show_main_page()
                gestionUsers()
            elif not authentication_status:
                with info:
                    st.error('Incorrect Password or username')
            else:
                with info:
                    st.warning('Please feed in your credentials')
        else:
            with info:
               st.warning('Username does not exist, Please Sign up')
    
#with headerSection:
if "authentication_status" not in st.session_state:
    show_login_page()
else:
    if st.session_state["authentication_status"]:
        show_logout_page()
        gestionUsers()
    else:
        show_login_page()
