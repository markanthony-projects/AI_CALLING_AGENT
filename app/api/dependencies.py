# get_db lives in app.core.database so that app.core.security can depend on it without
# core importing from the api layer. Re-exported here because the routes import it from
# this module and there is no value in churning every one of them.
from app.core.database import get_db

__all__ = ["get_db"]
