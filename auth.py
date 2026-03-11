import os
import jwt
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from werkzeug.security import generate_password_hash, check_password_hash
import database

# Router definition
router = APIRouter(prefix="/api/auth")

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"

# ============ Pydantic Models ============
class OTPRequest(BaseModel):
    email: str

class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    otp: str

class UserLogin(BaseModel):
    username: str
    password: str

# ============ Helper Functions ============
def send_email_smtp(email_to: str, otp: str):
    """Send OTP via Gmail SMTP (SSL port 465)."""
    sender_email = os.environ.get("SMTP_EMAIL", "")
    sender_password = os.environ.get("SMTP_PASSWORD", "")

    if not sender_email or not sender_password:
        # Fallback for local development if no SMTP credentials are set
        print(f"--- DEVELOPMENT MODE: Send OTP Email ---")
        print(f"To: {email_to}")
        print(f"OTP: {otp}")
        print(f"----------------------------------------")
        return

    subject = "AI News Verifier — Your Verification Code"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;background:#f9f9f9;padding:32px;border-radius:12px;">
      <h2 style="color:#4F46E5;margin-bottom:8px;">AI News Verifier</h2>
      <p style="color:#555;font-size:15px;">Your email verification code is:</p>
      <div style="font-size:40px;font-weight:bold;letter-spacing:10px;color:#1e1e2e;text-align:center;padding:20px 0;">{otp}</div>
      <p style="color:#888;font-size:13px;">This code expires in <strong>10 minutes</strong>. If you didn't request this, please ignore this email.</p>
    </div>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = email_to
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email_to, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        print(f"OTP for {email_to} is {otp}")

def create_access_token(user_id: str, username: str, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    if expires_delta is None:
        expires_delta = timedelta(hours=24)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {"sub": user_id, "username": username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        if token.startswith("Bearer "):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail={"error": "Invalid token"})
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={"error": "Token expired"})
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail={"error": "Invalid token"})

def get_current_user_id(authorization: Optional[str] = Header(None)):
    """Dependency to get current user ID from header"""
    if not authorization:
        raise HTTPException(status_code=401, detail={"error": "Missing authorization header"})
    return verify_token(authorization)

# ============ Routes ============
@router.post("/send-otp")
async def send_otp(request: OTPRequest, background_tasks: BackgroundTasks):
    """Generate and send OTP to the provided email."""
    email = request.email.strip().lower()
    
    # Strictly validate .gmail.com
    if not email.endswith("@gmail.com"):
        return JSONResponse(status_code=400, content={"error": "Only @gmail.com email addresses are allowed"})
    
    # Check if user already exists
    existing_user_by_email = database.db.users.find_one({"email": email})
    if existing_user_by_email:
         return JSONResponse(status_code=400, content={"error": "Email is already registered"})
         
    # Generate OTP
    otp = ''.join(random.choices(string.digits, k=6))
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    
    # Save to db
    database.save_otp(email, otp, expires_at)
    
    # Send email in background
    background_tasks.add_task(send_email_smtp, email, otp)
    
    return {"message": "OTP sent successfully"}

@router.post("/register")
async def register(user: UserRegister):
    """Register a new user."""
    username = user.username.strip()
    email = user.email.strip().lower()
    otp = user.otp.strip()
    
    if not username or not user.password:
        return JSONResponse(status_code=400, content={"error": "Username and password are required"})
    
    if len(username) < 3:
        return JSONResponse(status_code=400, content={"error": "Username must be at least 3 characters"})
    
    if len(user.password) < 6:
        return JSONResponse(status_code=400, content={"error": "Password must be at least 6 characters"})
        
    if not email.endswith("@gmail.com"):
        return JSONResponse(status_code=400, content={"error": "Only @gmail.com email addresses are allowed"})
    
    # Verify OTP
    if not database.verify_otp(email, otp):
        return JSONResponse(status_code=400, content={"error": "Invalid or expired OTP"})
    
    # Check if user exists
    existing_user = database.get_user_by_username(username)
    if existing_user:
        return JSONResponse(status_code=400, content={"error": "Username already exists"})
    
    existing_user_by_email = database.db.users.find_one({"email": email})
    if existing_user_by_email:
        return JSONResponse(status_code=400, content={"error": "Email is already registered"})
    
    # Create user
    try:
        # Hashing using werkzeug.security generate_password_hash
        password_hash = generate_password_hash(user.password)
        user_id = database.create_user(username, password_hash, email)
        
        # Generate tokens
        access_token = create_access_token(user_id, username)
        
        return {
            "message": "User registered successfully",
            "user": {
                "id": user_id,
                "username": username,
                "email": email
            },
            "access_token": access_token
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/login")
async def login(user: UserLogin):
    """Login and receive JWT token."""
    db_user = database.get_user_by_username(user.username)
    if not db_user or not check_password_hash(db_user["password"], user.password):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
    
    # Generate tokens
    access_token = create_access_token(db_user["id"], db_user["username"])
    
    return {
        "message": "Login successful",
        "user": {
            "id": db_user["id"],
            "username": db_user["username"],
            "email": db_user.get("email")
        },
        "access_token": access_token
    }

@router.get("/me")
async def get_current_user_info(user_id: str = Depends(get_current_user_id)):
    """Get current user info."""
    user = database.get_user_by_id(user_id)
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found"})
    
    # Get user stats
    stats = database.get_user_stats(user_id)
    
    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user.get("email"),
            "created_at": user["created_at"]
        },
        "stats": stats
    }

@router.post("/logout")
async def logout(user_id: str = Depends(get_current_user_id)):
    """Logout endpoint."""
    # Since we are using JWT, actual invalidation happens client-side by deleting the token.
    # We could implement a token blacklist here in the future if needed.
    return {"message": "Logged out successfully"}
