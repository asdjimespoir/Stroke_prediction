import re
import os
import smtplib
import datetime
from deta import Deta
import streamlit as st
from email import encoders
from dotenv import load_dotenv
import streamlit_authenticator as stauth
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


#Load the environment variables
load_dotenv(".env")
#DETA_KEY = os.getenv("DETA_KEY")
DETA_KEY = 'a0m78dwvhdm_1SDma2An8odFQf6Ti6QN2Nr9FxjztVbi'

deta = Deta(DETA_KEY)

db = deta.Base('FedIA')


def send_email_tab(user_email, name, pred, csv_file_path):
    # Configurer les détails du serveur SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'espoir.asdjimguel@gmail.com'
    smtp_password = 'nepw yceu emwg ttvd'

    # Construire le message
    msg = MIMEMultipart()
    msg['From'] = 'espoir.asdjimguel@gmail.com'
    msg['To'] = user_email
    msg['Subject'] = 'Le résultat de la prédiction'

    # Corps du message
    body = "Bonjour {}! \n\nLe patient risque l'AVC à : {:.3%}\nMerci de le prendre en charge dans un bref délai\n\nCordialement,\nL'équipe de l application de prédiction".format(name, pred)

    msg.attach(MIMEText(body, 'plain'))
    
    # Lisez et attachez le fichier en tant que pièce jointe
    with open(csv_file_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename(csv_file_path)}",
    )

    msg.attach(part)

    # Connexion au serveur SMTP
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        print('Email envoyé avec succès')
    except Exception as e:
        print(f'Erreur lors de l\'envoi de l\'email : {str(e)}')
        
def send_email_img(user_email, name, pred):
    # Configurer les détails du serveur SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'espoir.asdjimguel@gmail.com'
    smtp_password = 'nepw yceu emwg ttvd'

    # Construire le message
    msg = MIMEMultipart()
    msg['From'] = 'espoir.asdjimguel@gmail.com'
    msg['To'] = user_email
    msg['Subject'] = 'Alerte!!! Le résultat de la prédiction'

    # Corps du message
    body = "Bonjour {}! \n\nLe patient risque l'AVC à : {:.3%}\nMerci de le prendre en charge dans un bref délai\n\nCordialement,\nL'équipe de l'application de prédiction".format(name, pred)

    msg.attach(MIMEText(body, 'plain'))

    # Connexion au serveur SMTP
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        print('Email envoyé avec succès')
    except Exception as e:
        print(f'Erreur lors de l\'envoi de l\'email : {str(e)}')
        
def fetch_users():
    """
    Fetch Users
    :return Dictionary of Users:
    """
    users = db.fetch()
    return users.items

def insert_user(email, name, username, password, role):
    """
    Inserts Users into the DB
    :param email:
    :param name :
    :param username:
    :param password:
    :param role :
    :return User Upon successful Creation:
    """
    date_joined = str(datetime.datetime.now())

    return db.put({'key': email, 'name': name, 'username': username, 'password': password, 'role': role, 'date_joined': date_joined})

def update_user(new_email, new_name, new_username, new_password, new_role):
    """
    Inserts Users into the DB
    :param email:
    :param name :
    :param username:
    :param password:
    :param role :
    :return User Upon successful Creation:
    """
    date_joined = str(datetime.datetime.now())

    return db.update({'key': new_email, 'date_joined': date_joined, 'name': new_name, 'password': new_password, 'role': new_role, 'username': new_username})

def delete_user(email):
    """
    Supprime un utilisateur de la base de données Deta en utilisant son adresse e-mail.
    :param email: Adresse e-mail de l'utilisateur à supprimer.
    :return: True si la suppression réussit, False sinon.
    """
    # Utilisez la méthode 'delete' pour supprimer l'utilisateur avec la clé (adresse e-mail) spécifiée
    deleted = db.delete(email)

    # Vérifiez si la suppression a réussi
    if deleted:
        return True
    else:
        return False

def get_user_emails():
    """
    Fetch User Emails
    :return List of user emails:
    """
    users = db.fetch()
    emails = []
    for user in users.items:
        emails.append(user['key'])
    return emails

def get_names():
    """
    Fetch Usernames
    :return List of user usernames:
    """
    users = db.fetch()
    names = []
    for user in users.items:
        names.append(user['key'])
    return names

def get_usernames():
    """
    Fetch Usernames
    :return List of user usernames:
    """
    users = db.fetch()
    usernames = []
    for user in users.items:
        usernames.append(user['key'])
    return usernames

def validate_email(email):
    """
    Check Email Validity
    :param email:
    :return True if email is valid else False:
    """
    pattern = "^[a-zA-Z0-9-_.]+@[a-zA-Z0-9.-]+\.[a-z]{1,3}$" #tesQQ12@gmail.com

    if re.match(pattern, email):
        return True
    return False

def validate_username(username):
    """
    Checks Validity of userName
    :param username:
    :return True if username is valid else False:
    """

    pattern = "^[a-zA-Z0-9]*$"
    if re.match(pattern, username):
        return True
    return False

def sign_up():
    with st.form(key='signup', clear_on_submit=True):
        st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        name = st.text_input(':blue[Name]', placeholder='Enter Your Name')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        role = st.selectbox(':blue[Role]',("User", "Administrateur"), placeholder='Choice Role')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')
        
        #if email and name and username:
        if email:
            if validate_email(email):
                if email not in get_user_emails():
                    if validate_username(username):
                        if username not in get_usernames():
                            if len(username) >= 2:
                                if len(password1) >= 6:
                                    if password1 == password2:
                                        # Add User to DB
                                        hashed_password = stauth.Hasher([password2]).generate()
                                        insert_user(email, name, username, hashed_password[0],role)
                                        st.success('Account created successfully!!')
                                        st.balloons()
                                    else:
                                        st.warning('Passwords Do Not Match')
                                else:
                                    st.warning('Password is too Short')
                            else:
                                st.warning('Username Too short')
                        else:
                            st.warning('Username Already Exists')
                    else:
                        st.warning('Invalid Username')
                else:
                    st.warning('Email Already exists!!')
            else:
                st.warning('Invalid Email')
        #else:
        #    st.warning('Please fill in all required fields')
        btn1, bt2, btn3, btn4, btn5 = st.columns(5)

        with btn3:
            st.form_submit_button('Sign Up')

#sign_up()
