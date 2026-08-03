from dotenv import load_dotenv

load_dotenv()

from pydantic import Field, field_validator, model_validator
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

    # The dashboard runs in a browser, so it can never hold API_KEY — that key dials, and
    # dialing costs money. It authenticates as a named user instead and carries a session
    # in an httpOnly cookie, which script on the page cannot read.
    DASHBOARD_SESSION_SECRET: str = Field(default="", min_length=0)
    DASHBOARD_SESSION_TTL_SECONDS: int = 43_200  # 12h — one working day, then re-auth
    # Comma-separated. The dashboard is served from its own origin, so the API must name it
    # explicitly: credentialed CORS forbids "*", and a wrong entry here silently blocks login.
    DASHBOARD_CORS_ORIGINS: str = ""
    # Off only for plain-HTTP local dev. A Secure cookie is never sent over http://, so
    # leaving this on locally makes login appear to succeed and every later call 401.
    DASHBOARD_COOKIE_SECURE: bool = True

    # Explicit rather than derived from CORS, because "cross-origin" and "cross-site" are
    # different questions and only the second one decides this.
    #
    #   dashboard.homebble.in -> ai-calls.homebble.in   cross-origin, SAME SITE  -> lax works
    #   my-app.vercel.app     -> ai-calls.homebble.in   cross-site               -> needs none
    #
    # Prefer lax. SameSite=None makes the session a third-party cookie, which Safari and
    # Firefox already block by default — login there simply never sticks. Putting the Vercel
    # deployment on a homebble.in subdomain avoids the whole problem.
    DASHBOARD_COOKIE_SAMESITE: str = "lax"

    @field_validator("DASHBOARD_COOKIE_SAMESITE")
    @classmethod
    def samesite_is_a_real_value(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"lax", "none", "strict"}:
            raise ValueError("DASHBOARD_COOKIE_SAMESITE must be one of: lax, none, strict")
        return v

    @model_validator(mode="after")
    def samesite_none_requires_secure(self) -> "Settings":
        # Browsers reject SameSite=None without Secure outright: the cookie is never stored,
        # so login appears to succeed and every request after it is anonymous. Failing at
        # startup beats debugging that through a browser.
        if self.DASHBOARD_COOKIE_SAMESITE == "none" and not self.DASHBOARD_COOKIE_SECURE:
            raise ValueError(
                "DASHBOARD_COOKIE_SAMESITE=none requires DASHBOARD_COOKIE_SECURE=true; "
                "browsers discard a SameSite=None cookie that is not Secure"
            )
        return self

    @field_validator("DASHBOARD_SESSION_SECRET")
    @classmethod
    def session_secret_is_strong_enough(cls, v: str) -> str:
        # Empty disables the dashboard entirely, which is a valid deployment. A short secret
        # is not: it would sign forgeable sessions for an interface that can spend money.
        if v and len(v) < 32:
            raise ValueError("DASHBOARD_SESSION_SECRET must be at least 32 characters")
        return v

    @property
    def dashboard_enabled(self) -> bool:
        return bool(self.DASHBOARD_SESSION_SECRET)

    @property
    def dashboard_cors_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.DASHBOARD_CORS_ORIGINS.split(",") if o.strip()]

    VOBIZ_AUTH_ID: str = ""
    VOBIZ_AUTH_TOKEN: str = ""
    VOBIZ_PHONE_NUMBER: str = ""
    WEBHOOK_BASE_URL: str = ""

    # Country code applied to local-format lead numbers (91 = India)
    DEFAULT_COUNTRY_CODE: str = "91"

    # Vobiz bills at dial time, so a leaked API key spends money whether or not the audio
    # ever connects. MAX_CONCURRENT_CALLS only caps streams; these cap the dialing itself.
    DIAL_MAX_PER_MINUTE: int = Field(default=30, ge=1)
    DIAL_MAX_PER_DAY: int = Field(default=500, ge=1)

    # Streams accepted at once. Each one runs Silero VAD plus audio resampling on the CPU,
    # so this has to match the droplet: roughly two calls per vCPU before audio starts to
    # break up. Raising it past what the host can carry degrades every call in progress
    # rather than rejecting the extra one.
    MAX_CONCURRENT_CALLS: int = Field(default=4, ge=1)

    # Barge-in sensitivity. Pipecat defaults are 0.7 / 0.6; running at 0.5 / 0.1 let PSTN
    # line noise clear the bar and cut the agent off mid-sentence, leaving callers saying
    # "Hello?" into a line that had gone quiet. Tunable without a code change so these can
    # be measured against real calls.
    VAD_CONFIDENCE: float = 0.7
    VAD_MIN_VOLUME: float = 0.4

    # Must stay below the STT's p99 transcript latency. Pipecat waits
    # `max(0, stt_p99 - stop_secs)` for transcripts before its turn analyzer decides the
    # turn is over, so raising this past Deepgram's 0.35s collapses that window to zero and
    # the analyzer runs blind. Running at 0.6 to stop mid-sentence splits did exactly that,
    # and added 400ms to every turn on top. Split turns are the analyzer's job — the lever
    # for them is user_turn_stop_timeout on LLMUserAggregatorParams, not this.
    VAD_STOP_SECS: float = 0.2

    # Words the prospect must say to cut the agent off while it is speaking. VAD alone
    # treated the "Hello?" on pickup as a barge-in and killed the opening line 0.7s in, so
    # the prospect never heard who was calling and asked "who are you?" two turns later.
    # Below this the agent keeps talking, which is also what should happen for "haan",
    # "hmm" and "achha" — those are the prospect listening, not interrupting. Once the
    # agent stops speaking a single word starts their turn, so replies stay instant.
    INTERRUPT_MIN_WORDS: int = Field(default=3, ge=1)

    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    # How long after the prospect stops making sound their turn stays open for them to
    # carry on. VAD_STOP_SECS cannot do this job — it must stay under the STT's p99 or turn
    # detection runs blind, so at 0.2s a normal mid-sentence breath ended the turn. "Maybe
    # around in 2" and "months." arrived as two turns, each costing a full LLM request, and
    # the agent answered half a sentence. Both observed gaps were under 0.75s.
    #
    # This is a straight trade: it adds its own value to every turn's response time, and
    # buys back a duplicate request plus an answer to half a question. Lower it only with
    # call logs showing turns are no longer splitting.
    TURN_SETTLE_SECS: float = Field(default=0.8, gt=0)

    OPENAI_API_KEY: str
    SARVAM_API_KEY: str
    SARVAM_VOICE_ID: str = "simran"
    GROQ_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""

    CEREBRAS_API_KEY: str = ""

    # Which provider serves the calls. Deliberately configuration and not a class name:
    # Groq deprecated llama-3.3-70b-versatile with six weeks' notice and closed its
    # self-serve paid tier, so the next such notice should cost an env change rather than a
    # release. Any endpoint speaking the OpenAI wire format works.
    #
    # gemma-4-31b on Cerebras was chosen by replaying a real call against every candidate
    # and scoring the rules live calls had actually broken, not by tokens per second:
    #
    #   gemma-4-31b    503ms TTFB   end_call via the tool channel   name in 6 replies of 6
    #   gpt-oss-120b   559ms        tool                            5/6, and it ran a turn
    #                                                               behind on the opening
    #   zai-glm-4.7   1446ms        tool                            5/6, 5.8s worst case
    #
    # For comparison, Groq's llama-3.3 ran at 400ms but wrote end_call into its spoken text
    # on essentially every call, and its free tier allows 12,000 tokens/minute against a
    # 3,400-token request — under three turns a minute for a conversation that needs ten.
    LLM_PROVIDER_NAME: str = "cerebras"
    LLM_BASE_URL: str = "https://api.cerebras.ai/v1"
    # Blank resolves to the provider's own key below, so a deployment that predates this
    # setting keeps working on whichever key it already had.
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemma-4-31b"

    @property
    def llm_api_key(self) -> str:
        """The key for whichever provider LLM_BASE_URL points at.

        Resolved per provider rather than by a single fallback: a blanket
        `LLM_API_KEY or GROQ_API_KEY` would send a Groq key to Cerebras the moment the
        default endpoint changed, and the failure would arrive as a dropped call rather
        than as a configuration error.
        """
        if self.LLM_API_KEY:
            return self.LLM_API_KEY
        if "cerebras.ai" in self.LLM_BASE_URL:
            return self.CEREBRAS_API_KEY
        if "groq.com" in self.LLM_BASE_URL:
            return self.GROQ_API_KEY
        if "openai.com" in self.LLM_BASE_URL:
            return self.OPENAI_API_KEY
        return ""

    @model_validator(mode="after")
    def the_llm_has_a_key(self) -> "Settings":
        # Without this the first symptom is a caller hearing silence: the greeting is
        # queued, the completion 401s, and the recovery path apologises on our behalf.
        # A process that cannot make an LLM call should not accept a websocket.
        if not self.llm_api_key:
            raise ValueError(
                f"No API key for the configured LLM at {self.LLM_BASE_URL}. "
                f"Set LLM_API_KEY, or the provider's own key "
                f"(CEREBRAS_API_KEY / GROQ_API_KEY / OPENAI_API_KEY)."
            )
        return self

    # A second OpenAI-wire-format provider to finish a turn Groq refuses on a rate limit.
    # Off unless both the key and the model are set: a half-configured fallback looks like
    # insurance and fails at the one moment it is needed. Defaults to OpenAI's endpoint,
    # so LLM_FALLBACK_API_KEY plus LLM_FALLBACK_MODEL=gpt-4o-mini is enough to enable it.
    # Roughly one request's worth of tokens. Below this the account cannot answer the
    # opening turn, so a dial placed now bills for the carrier leg and rings a real person
    # who then hears silence. Set to 0 to dial regardless of the LLM budget.
    LLM_MIN_TOKENS_TO_DIAL: int = Field(default=4000, ge=0)

    LLM_FALLBACK_NAME: str = "openai"
    LLM_FALLBACK_API_KEY: str = ""
    LLM_FALLBACK_BASE_URL: str = "https://api.openai.com/v1"
    LLM_FALLBACK_MODEL: str = ""

    @property
    def llm_fallback_enabled(self) -> bool:
        return bool(self.LLM_FALLBACK_API_KEY and self.LLM_FALLBACK_MODEL)

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def format_database_url(cls, v: str) -> str:
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
