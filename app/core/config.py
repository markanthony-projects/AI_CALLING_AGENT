from dotenv import load_dotenv

load_dotenv()

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Indian Real Estate Sales Voice Agent"

    # Defaults to on: production gets auth by simply not setting this
    AUTH_ENABLED: bool = True

    # /docs and /openapi.json sit outside the API-key dependency, so on a public host they
    # publish the full route list and request shapes to anyone. Off unless asked for.
    DOCS_ENABLED: bool = False
    API_KEY: str = Field(min_length=32)
    CALL_TOKEN_SECRET: str = Field(min_length=32)
    CALL_TOKEN_TTL_SECONDS: int = 900

    VOBIZ_AUTH_ID: str = ""
    VOBIZ_AUTH_TOKEN: str = ""
    VOBIZ_PHONE_NUMBER: str = ""
    WEBHOOK_BASE_URL: str = ""

    # Country code applied to local-format lead numbers (91 = India)
    DEFAULT_COUNTRY_CODE: str = "91"

    # Vobiz bills at dial time, so a leaked API key spends money whether or not the audio
    # ever connects. MAX_CALLS only caps concurrent streams; these cap the dialing itself.
    DIAL_MAX_PER_MINUTE: int = Field(default=30, ge=1)
    DIAL_MAX_PER_DAY: int = Field(default=500, ge=1)

    # Barge-in sensitivity. Pipecat defaults are 0.7 / 0.6; running at 0.5 / 0.1 let PSTN
    # line noise clear the bar and cut the agent off mid-sentence, leaving callers saying
    # "Hello?" into a line that had gone quiet. Tunable without a code change so these can
    # be measured against real calls.
    VAD_CONFIDENCE: float = 0.7
    VAD_MIN_VOLUME: float = 0.4

    # How long a caller stays silent before their turn is treated as finished. Pipecat
    # defaults to 0.2s, which suits a headset; over PSTN a mid-sentence breath ended the
    # turn, so "Yeah sure. Sunday works for me." arrived as two turns and ran two separate
    # LLM inferences — one asked for the time, the other hung up. Measured pauses on real
    # calls: 234ms, 328ms, 708ms. This is added wait on every turn, so it trades directly
    # against voice-to-voice latency.
    VAD_STOP_SECS: float = 0.6

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
