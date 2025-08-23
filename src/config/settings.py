"""
프로젝트 설정 관리
환경변수 로드 및 설정 검증
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv


# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# .env 파일 로드
load_dotenv(BASE_DIR / '.env')


class ExchangeSettings(BaseSettings):
    """거래소 API 설정"""
    upbit_access_key: str = Field(..., env='UPBIT_ACCESS_KEY')
    upbit_secret_key: str = Field(..., env='UPBIT_SECRET_KEY')
    binance_api_key: str = Field(..., env='BINANCE_API_KEY')
    binance_secret_key: str = Field(..., env='BINANCE_SECRET_KEY')
    
    class Config:
        env_file = '.env'
        case_sensitive = False


class TradingSettings(BaseSettings):
    """거래 설정"""
    capital_per_exchange: int = Field(20000000, env='CAPITAL_PER_EXCHANGE')
    max_position_size_percent: float = Field(1.0, env='MAX_POSITION_SIZE_PERCENT')
    kimchi_premium_entry_threshold: float = Field(4.0, env='KIMCHI_PREMIUM_ENTRY_THRESHOLD')
    kimchi_premium_exit_threshold: float = Field(2.0, env='KIMCHI_PREMIUM_EXIT_THRESHOLD')
    
    # 리스크 관리
    max_exposure_percent: float = 30.0  # 최대 노출 30%
    stop_loss_atr_multiplier: float = 2.0  # ATR * 2 손절
    
    @validator('max_position_size_percent')
    def validate_position_size(cls, v):
        if not 0 < v <= 5:
            raise ValueError('Position size must be between 0 and 5 percent')
        return v
    
    class Config:
        env_file = '.env'


class MonitoringSettings(BaseSettings):
    """모니터링 설정"""
    slack_webhook_url: Optional[str] = Field(None, env='SLACK_WEBHOOK_URL')
    discord_webhook_url: Optional[str] = Field(None, env='DISCORD_WEBHOOK_URL')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    
    class Config:
        env_file = '.env'


class DataSettings(BaseSettings):
    """데이터 설정"""
    data_dir: Path = BASE_DIR / 'data'
    raw_data_dir: Path = BASE_DIR / 'data' / 'raw'
    processed_data_dir: Path = BASE_DIR / 'data' / 'processed'
    cache_dir: Path = BASE_DIR / 'data' / 'cache'
    
    # 데이터 수집 빈도 (초)
    price_update_interval: int = 60  # 1분
    orderbook_update_interval: int = 15  # 15초
    exchange_rate_update_interval: int = 60  # 1분
    
    # 데이터 보관 기간 (일)
    raw_data_retention_days: int = 30
    processed_data_retention_days: int = 90
    
    def ensure_directories(self):
        """데이터 디렉토리 생성"""
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ModelSettings(BaseSettings):
    """모델 설정"""
    model_dir: Path = BASE_DIR / 'models'
    lstm_model_dir: Path = BASE_DIR / 'models' / 'lstm'
    xgboost_model_dir: Path = BASE_DIR / 'models' / 'xgboost'
    rl_model_dir: Path = BASE_DIR / 'models' / 'rl'
    
    # LSTM 파라미터
    lstm_sequence_length: int = 60  # 60분 시퀀스
    lstm_hidden_units: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.2
    
    # XGBoost 파라미터
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.01
    xgb_n_estimators: int = 500
    
    # RL 파라미터
    rl_learning_rate: float = 3e-4
    rl_batch_size: int = 64
    
    def ensure_directories(self):
        """모델 디렉토리 생성"""
        for dir_path in [self.lstm_model_dir, self.xgboost_model_dir, self.rl_model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """통합 설정"""
    environment: str = Field('development', env='ENVIRONMENT')
    debug_mode: bool = Field(True, env='DEBUG_MODE')
    
    # 하위 설정들
    exchange: ExchangeSettings = ExchangeSettings()
    trading: TradingSettings = TradingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    
    class Config:
        env_file = '.env'
        case_sensitive = False
    
    def initialize(self):
        """초기화 - 필요한 디렉토리 생성"""
        self.data.ensure_directories()
        self.model.ensure_directories()
        
        # 로그 디렉토리 생성
        log_dir = BASE_DIR / 'logs'
        log_dir.mkdir(exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == 'development'


# 전역 설정 인스턴스
settings = Settings()

# 초기화
settings.initialize()