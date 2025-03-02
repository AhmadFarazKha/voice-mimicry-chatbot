from pydantic import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    DATABASE_URL: str = "sqlite:///./test.db"
    MODEL_STORAGE_PATH: str = "./models"

    class Config:
        env_file = ".env"

settings = Settings()