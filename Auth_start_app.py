import streamlit as st
from streamlit_lottie import st_lottie
import requests
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import auth
import json
# from demo_app import app
from main import main
from datetime import datetime
from streamlit_cookies_manager import EncryptedCookieManager
import pytz
import os
from dotenv import load_dotenv

# Changed the original File at location - "C:\Users\AADITYA\anaconda3\envs\newenv\Lib\site-packages\streamlit_cookies_manager\encrypted_cookie_manager.py" 
# because it was giving st.cache deprecation warning. So made change from @st.cache--> @st.cache_data, now warning has stopped occuring at beginning of execution.


load_dotenv()

FIREBASE_API_key = os.getenv('firebase_key')
cookie_key = os.getenv('cookie_key')  # Replace with your own secret key

cookies = EncryptedCookieManager(prefix="my_app_", password=cookie_key)
if not cookies.ready():
    st.stop()

if not firebase_admin._apps:
        cred = credentials.Certificate("authentication-page-c6bf1-7a3ee3b12cb0.json")
        firebase_admin.initialize_app(cred)

db = firestore.client()

if 'final_access' not in st.session_state:
        st.session_state.final_access = False


def authentication():
    
    def load_css(file_path):
        with open(file_path, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)   
    load_css("Authpage_styles.css")


    # st.markdown("<h1 style='text-align: center;'>Welcome to our Application</h1>", unsafe_allow_html=True)
    def get_current_time_ist():
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        return current_time.strftime('%Y-%m-%d, %H:%M:%S')  # Format: YYYY-MM-DD, HH:MM:SS


    # Function to load Lottie animation
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    # Load Lottie animation
    lottie_coding = load_lottieurl('https://lottie.host/91c62d14-f11f-4938-9873-39272608b35a/bDCN5zDhHZ.json')
    # Center the Lottie animation
    if lottie_coding:
        st_lottie(lottie_coding, height=300, width=650, key="lottie")

    # Centered Text
    st.markdown("<h6 style='text-align: center;'>Select the authentication method</h6>", unsafe_allow_html=True)





    # =======================================================            =================================================================#
                                                              #  MAIN  #
    # =======================================================            =================================================================#


    # Session state initialization
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'reset_password_flow' not in st.session_state:
        st.session_state.reset_password_flow = False
    if 'reset_message' not in st.session_state:
        st.session_state.reset_message = ""
    
    

    _, login_col, signup_col, __ = st.columns([0.345, 0.12, 0.15, 0.35])
    with login_col:
        if st.button('Login'):
            st.session_state.show_login = True
            st.session_state.show_signup = False
            st.session_state.reset_password_flow = False
    with signup_col:
        if st.button('Sign Up'):
            st.session_state.show_signup = True
            st.session_state.show_login = False
            st.session_state.reset_password_flow = False



    def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
        st.session_state.success_signup = False

        try:
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
            payload = {
                "email": email,       # these payload keys are predefined keys used during requesting firebase authentication.These are not used to directly display anywhere in DB. 
                "password": password,       # other keys include: displayName,idToken, photoURL
                "returnSecureToken": return_secure_token        
            }
            if username:
                payload["displayName"] = username

            payload = json.dumps(payload)
            # requesting for signing-up
            r = requests.post(
                rest_api_url,
                params={"key": FIREBASE_API_key},
                data=payload,
            )

            # Handle Firebase response
            user_data = r.json()
            if "email" in user_data:
                user_id = user_data["localId"]  # Unique user ID from Firebase Authentication
                signup_date_time = get_current_time_ist()
                db.collection("users").document(user_id).set(
                    {"username": username, "email": email, 'Activate':False,
                     "SignUp Date/Time": signup_date_time, "Last_Login": ' '}  #Firebase Authentication handles passwords securely and separately from user profile data so no need to display in user profile.   
                )
                st.session_state.success_signup = True
                return user_data["email"], st.session_state.success_signup
            else:
                # Custom error handling. If the json response is successful then it executes above 'if' condition, else if its returned with error, that means email is already registered.
                error_message = user_data.get("error", {}).get("message", "Unknown error occurred.")
                if error_message == "EMAIL_EXISTS":
                    st.error("This email is already registered. Please try logging in.")
                else:
                    st.error(f"Sign-up failed: {error_message}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def login_with_email_and_password(email=None, password=None, return_secure_token=True):
        try:
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": return_secure_token
            }
            payload = json.dumps(payload)
            r = requests.post(rest_api_url, params={"key": FIREBASE_API_key}, data=payload)
            data = r.json()
            if 'email' in data:
                user_id = data.get('localId')
                user_doc = db.collection('users').document(user_id).get()

                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    if user_data.get('Activate',False):  # Checks if the Activate field in the user document is True. If missing, then sets default value to False.
                        last_login_logout_date_time = get_current_time_ist()
                        db.collection('users').document(user_id).update(
                            {
                                "Last_Login": last_login_logout_date_time  # Update last login time
                            }
                        )
                        st.session_state.final_access = True
                        return {
                            'email': data['email'],
                            'username': user_data.get('username', 'User')
                        } 
                    else:
                        st.warning("Your account is not activated yet. Please contact the administrator.")
                        return None
                else:
                    st.error("User data not found in the database.")
                    return None
            else:
                st.warning(data.get('error', {}).get('message', 'Login failed.'))
                return None
        except Exception as e:
            st.warning(f'Login failed: {e}')
            return None

    def reset_password(email):
        try:
            # Check if the account exists and is activated
            user_docs = db.collection("users").where("email", "==", email).get()
            if not user_docs:
                return False, "No account found with this email."
            
            user_data = user_docs[0].to_dict()  # Assuming email is unique and fetch the first document
            if not user_data.get("Activate", False):  # Check if the 'Activate' field is True
                return False, "Your account is not activated yet. Please contact the administrator."

            # Proceed with reset password if account is activated
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
            payload = {"email": email, "requestType": "PASSWORD_RESET"}
            r = requests.post(
                rest_api_url, 
                params={"key": FIREBASE_API_key}, 
                data=json.dumps(payload)
            )
            if r.status_code == 200:
                return True, "Password reset email sent successfully. Check your inbox!"
            else:
                return False, r.json().get("error", {}).get("message", "Unknown error occurred.")
        except Exception as e:
            return False, f"An error occurred: {str(e)}"



    # SignUp Form
    if st.session_state.show_signup:
        st.markdown("<h3 style='text-align: center;'>Sign-Up</h3>", unsafe_allow_html=True)
        username = st.text_input("Enter your username")
        email = st.text_input('Enter New Email:', key='signup_email')
        password = st.text_input('Enter New Password:', type='password', key='signup_password')
        if st.button('Create my account'):
            if not username:
                st.warning("Username is required to create an account.")
            else:
                sign_up_with_email_and_password(email, password, username)
                if st.session_state.success_signup:
                    st.success('Account created successfully! Please contact the administrator to get your account activated before logging in!')



    # Login Form
    if st.session_state.show_login:
        st.markdown("<h3 style='text-align: center;'>Log-in</h3>", unsafe_allow_html=True)
        email = st.text_input('Enter Email:', key='email')
        password = st.text_input('Enter Password:', type='password', key='password')

        if st.button('Login', key="login_button"):
            user_info = login_with_email_and_password(email, password)
            cookies["auth_token"] = "user_token" 
            cookies['username_cookie'] = user_info['username']
            cookies.save()
            if user_info:                
                st.session_state.show_login = False
                st.rerun()
                return st.session_state.final_access
                            
            else:
                st.session_state.reset_password_flow = False

        # Forgot Password Flow
        if not st.session_state.reset_password_flow:
            if st.button('Forgot Password?'):
                st.session_state.reset_password_flow = True

        if st.session_state.reset_password_flow:
            st.markdown("<h4 style='text-align: center;'>Reset Password</h4>", unsafe_allow_html=True)
            reset_email = st.text_input('Enter your email for password reset:')
            if st.button('Reset Password'):
                if reset_email:
                    success, msg = reset_password(reset_email)
                    st.session_state.reset_message = msg
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter your email to reset your password.")



if "auth_token" in cookies and cookies["auth_token"]:
    st.session_state.final_access = True
else:
    st.session_state.final_access = False

    
if st.session_state.final_access:
    username = cookies['username_cookie']
    st.sidebar.markdown(f"<h1 style='text-align: center; color:  #81538c;'>Welcome {username}!</h2>", unsafe_allow_html=True,)  
    # st.sidebar.header(f'Welcome {username}')  

    col1, col2 = st.columns([0.8, 0.2])  # Adjust column width for placement
    with col2:
        if st.button(f"Logout"):
            cookies['auth_token'] = ""
            cookies['username_cookie']
            # cookies("auth_token", "")
            cookies.save()
            st.session_state.final_access=False
            # st.session_state.clear()  # Clears all session state variables
            st.rerun()
    main()  # Render the main app page
else:
    authentication() 

# if __name__=='__main__':            # uncomment when executing test.py
#     authentication()