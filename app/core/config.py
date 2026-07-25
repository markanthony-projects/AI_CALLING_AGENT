from dotenv import load_dotenv
import os

load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "Indian Real Estate Sales Voice Agent"
    
    # Vobiz Credentials
    VOBIZ_AUTH_ID: str = os.getenv("VOBIZ_AUTH_ID", "")
    VOBIZ_AUTH_TOKEN: str = os.getenv("VOBIZ_AUTH_TOKEN", "")
    VOBIZ_PHONE_NUMBER: str = os.getenv("VOBIZ_PHONE_NUMBER", "")
    WEBHOOK_BASE_URL: str = os.getenv("WEBHOOK_BASE_URL", "")
    
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"
    
    OPENAI_API_KEY: str
    SARVAM_API_KEY: str
    SARVAM_VOICE_ID: str = "simran"
    GROQ_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def format_database_url(cls, v: str) -> str:
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
