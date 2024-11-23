from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Get Supabase credentials
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

# Initialize Supabase client with URL validation
if not supabase_url or not supabase_url.startswith(('http://', 'https://')):
    raise ValueError("Invalid SUPABASE_URL. Must start with http:// or https://")

try:
    # Create Supabase client with minimal configuration
    supabase = create_client(supabase_url, supabase_key)
except Exception as e:
    print(f"Error initializing Supabase client: {str(e)}")
    raise

def track_user_registration(email, name):
    try:
        data = {
            'email': email,
            'name': name,
            'registered_at': datetime.utcnow().isoformat()
        }
        result = supabase.table('user_registrations').insert(data).execute()
        return result.data
    except Exception as e:
        print(f"Error tracking user registration: {str(e)}")
        return None

def get_registration_stats():
    try:
        result = supabase.table('user_registrations').select('*').execute()
        return result.data
    except Exception as e:
        print(f"Error getting registration stats: {str(e)}")
        return []
