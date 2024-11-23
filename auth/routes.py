from flask import Blueprint, render_template, jsonify, request, current_app
import requests
import os
from .clerk_auth import get_user_id
from database.supabase_db import track_user_registration

auth_bp = Blueprint('auth', __name__)

def get_user_details(user_id):
    """Get user details from Clerk's API"""
    try:
        headers = {
            'Authorization': f'Bearer {os.getenv("CLERK_SECRET_KEY")}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            f'https://api.clerk.com/v1/users/{user_id}',
            headers=headers
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error getting user details: {str(e)}")
        return None

@auth_bp.route('/sign-in')
def sign_in():
    # Pass Clerk configuration to template
    return render_template('auth/sign-in.html',
                         clerk_publishable_key=current_app.config['CLERK_PUBLISHABLE_KEY'],
                         clerk_frontend_api=current_app.config['CLERK_FRONTEND_API'])

@auth_bp.route('/sign-up')
def sign_up():
    # Pass Clerk configuration to template
    return render_template('auth/sign-up.html',
                         clerk_publishable_key=current_app.config['CLERK_PUBLISHABLE_KEY'],
                         clerk_frontend_api=current_app.config['CLERK_FRONTEND_API'])

@auth_bp.route('/user', methods=['GET'])
def get_user():
    try:
        user_id = get_user_id()
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401

        user_details = get_user_details(user_id)
        if not user_details:
            return jsonify({'error': 'Failed to get user details'}), 500

        # Extract email and name from user details
        email = user_details.get('email_addresses', [{}])[0].get('email_address')
        first_name = user_details.get('first_name', '')
        last_name = user_details.get('last_name', '')
        name = f"{first_name} {last_name}".strip() or email

        # Track user registration in Supabase
        if email and name:
            track_user_registration(user_id, email, name)

        return jsonify({
            'id': user_id,
            'email': email,
            'name': name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
