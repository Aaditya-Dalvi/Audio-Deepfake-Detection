import firebase_admin
from firebase_admin import auth
from firebase_admin import credentials

# Initialize the Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("authentication-page-c6bf1-7a3ee3b12cb0.json")
    firebase_admin.initialize_app(cred)

def delete_user_by_email(email):
    try:
        user = auth.get_user_by_email(email)  # Fetch the user by email
        auth.delete_user(user.uid)  # Delete the user by UID
        print(f"User with email {email} deleted successfully.")
    except firebase_admin.auth.UserNotFoundError:
        print(f"No user found with email {email}.")
    except Exception as e:
        print(f"Error deleting user: {str(e)}")

# Replace with the email you want to delete
email_to_delete = "testusernew@gmail.com"
delete_user_by_email(email_to_delete)
