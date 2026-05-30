# auth.py
# Optional API key authentication for the FastAPI backend.
#
# How it works:
#   - If the API_KEY environment variable is set, all protected endpoints
#     require an Authorization header: "Bearer <key>"
#   - If API_KEY is NOT set, authentication is disabled entirely —
#     perfect for local development, no friction
#   - The health check endpoint (/) is always public
#
# This approach keeps things simple: one environment variable controls
# whether auth is on or off. No user database, no OAuth, no tokens.
# For a privacy-first local product, this is exactly right.

import logging
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import API_KEY

logger = logging.getLogger(__name__)

# HTTPBearer extracts the token from "Authorization: Bearer <token>"
# auto_error=False so we can handle missing headers ourselves
_security = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(_security),
):
    """
    FastAPI dependency that enforces API key authentication.

    - If API_KEY is not configured: auth is disabled, everything passes
    - If API_KEY is configured: the request must include a valid Bearer token
    """
    if not API_KEY:
        # auth not configured — allow everything
        return None

    if not credentials:
        logger.warning("Rejected request: missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide an API key via "
                   "the Authorization header: 'Bearer <your-api-key>'",
        )

    if credentials.credentials != API_KEY:
        logger.warning("Rejected request: invalid API key")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return credentials.credentials
