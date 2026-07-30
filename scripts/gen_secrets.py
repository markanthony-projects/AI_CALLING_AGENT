"""Generate the two self-issued secrets and print them once.

API_KEY and CALL_TOKEN_SECRET are not obtained from any provider — they are values you
invent. Run this on the machine that will hold them and paste the output into the server's
env file; piping it anywhere that keeps history defeats the point.

    python scripts/gen_secrets.py
"""

import secrets

if __name__ == "__main__":
    print("# Paste into the server env file. Generated locally, never committed.")
    print(f"API_KEY={secrets.token_urlsafe(32)}")
    print(f"CALL_TOKEN_SECRET={secrets.token_urlsafe(32)}")
    print()
    print("# API_KEY is shared with whatever calls your API (sent as the X-API-Key header).")
    print("# CALL_TOKEN_SECRET never leaves the server. Changing it invalidates in-flight calls.")
