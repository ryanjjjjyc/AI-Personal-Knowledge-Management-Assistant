import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256" # HS = HMAC + SHA, 256 = SHA-256 hash. Full name: HMAC-SHA256
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ["ADMIN_PASSWORD_HASH"]  # bcrypt hash string

# Creates a FastAPI security dependency. It expects requests to include a bearer token
# oauth2_scheme behaves like a function that takes a request and returns a string (or raises 401)
# e.g.
# GET /admin/dashboard HTTP/1.1
# Host: example.com
# Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMiLCJyb2xlIjoiYWRtaW4ifQ.abc123signature
# "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMiLCJyb2xlIjoiYWRtaW4ifQ.abc123signature"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        # Converts both the plaintext password and stored hash from str to bytes, then compares them using bcrypt.checkpw
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash, or password > 72 bytes.
        return False


def authenticate_admin(username: str, password: str) -> bool:
    # Always run a check even when the username is wrong, so the timing
    # of "wrong user" and "wrong password" look similar.
    if username != ADMIN_USERNAME:
        # String to bytes for the specific text "dummy" as the username already does not match the admin username
        bcrypt.checkpw(b"dummy", ADMIN_PASSWORD_HASH.encode("utf-8")) 
        return False
    return verify_password(password, ADMIN_PASSWORD_HASH)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token. Adds the expire time and encode it back.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_admin(token: str = Depends(oauth2_scheme)) -> dict:
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise creds_exc

    username = payload.get("sub")
    role = payload.get("role")
    if not username or role != "admin":
        raise creds_exc
    return {"username": username, "role": role}