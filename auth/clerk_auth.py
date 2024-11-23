from functools import wraps
from flask import request, jsonify, current_app, redirect, url_for
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def verify_token(token):
    """Verify the session token with Clerk's backend API"""
    try:
        headers = {
            'Authorization': f'Bearer {os.getenv("CLERK_SECRET_KEY")}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            'https://api.clerk.com/v1/sessions/verify',
            headers=headers,
            params={'session_token': token}
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Token verification error: {str(e)}")
        return None

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if this is an API request
        is_api_request = request.headers.get('Accept', '').startswith('application/json')
        
        # Get token from Authorization header or session cookie
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        else:
            token = request.cookies.get('__session')

        if not token:
            if is_api_request:
                return jsonify({'error': 'No token provided'}), 401
            return redirect(url_for('auth.sign_in'))

        # Verify token
        session = verify_token(token)
        if not session:
            if is_api_request:
                return jsonify({'error': 'Invalid token'}), 401
            return redirect(url_for('auth.sign_in'))

        return f(*args, **kwargs)
    return decorated_function

def get_user_id():
    """Get the user ID from the current session"""
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    else:
        # Try session cookie
        token = request.cookies.get('__session')
    
    if not token:
        return None
    
    session = verify_token(token)
    return session.get('id') if session else None
