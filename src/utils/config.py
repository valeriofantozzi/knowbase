"""
Configuration Module

Manages application configuration with environment variable support and validation.
Includes model selection and validation for multi-model embedding support.
"""

import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from dotenv import load_dotenv


class Config:
    """Application configuration manager with validation."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()  # Try default locations
        
        # Model Configuration
        self.MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5")
        self.MODEL_CACHE_DIR = os.path.expanduser(
            os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface")
        )
        
        # Processing Configuration
        # Batch size will be auto-optimized if not explicitly set
        batch_size_env = os.getenv("BATCH_SIZE")
        if batch_size_env:
            self.BATCH_SIZE = self._get_int("BATCH_SIZE", 128)
        else:
            # Auto-optimize batch size based on hardware
            try:
                from .performance_optimizer import get_performance_optimizer
                optimizer = get_performance_optimizer()
                self.BATCH_SIZE = optimizer.get_optimal_batch_size()
            except Exception:
                # Fallback to default
                self.BATCH_SIZE = 128
        self.CHUNK_SIZE = self._get_int("CHUNK_SIZE", 300)
        self.CHUNK_OVERLAP = self._get_int("CHUNK_OVERLAP", 60)
        self.MIN_CHUNK_SIZE = self._get_int("MIN_CHUNK_SIZE", 50)
        
        # Vector Database Configuration
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "subtitle_embeddings")
        
        # Device Configuration
        device = os.getenv("DEVICE", "auto").lower()
        if device == "auto":
            # Use hardware detector for auto-detection (supports MPS)
            try:
                from .hardware_detector import get_hardware_detector
                hardware_detector = get_hardware_detector()
                self.DEVICE = hardware_detector.get_recommended_device()
            except Exception:
                # Fallback to basic detection
                import torch
                if torch.cuda.is_available():
                    self.DEVICE = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.DEVICE = "mps"
                else:
                    self.DEVICE = "cpu"
        elif device in ["cpu", "cuda", "mps"]:
            self.DEVICE = device
        else:
            raise ValueError(f"Invalid DEVICE value: {device}. Must be 'auto', 'cpu', 'cuda', or 'mps'")
        
        self.CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        
        # Logging Configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {log_level}. Must be one of {valid_log_levels}"
            )
        self.LOG_LEVEL = log_level
        
        log_file = os.getenv("LOG_FILE", "./logs/app.log")
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE = str(log_path)
        
        # Performance Configuration
        # Auto-optimize workers if not explicitly set
        workers_env = os.getenv("MAX_WORKERS")
        if workers_env:
            self.MAX_WORKERS = self._get_int("MAX_WORKERS", os.cpu_count() or 1)
        else:
            # Auto-optimize based on hardware
            try:
                from .performance_optimizer import get_performance_optimizer
                optimizer = get_performance_optimizer()
                self.MAX_WORKERS = optimizer.get_optimal_workers("cpu_bound")
            except Exception:
                # Fallback to CPU count
                self.MAX_WORKERS = os.cpu_count() or 1
        self.ENABLE_CHECKPOINTING = self._get_bool("ENABLE_CHECKPOINTING", True)
        self.CHECKPOINT_INTERVAL = self._get_int("CHECKPOINT_INTERVAL", 1000)
        
        # Validate configuration
        self._validate()

        # Validate model configuration
        self._validate_model()
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value for {key}: {value}")
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        value_lower = value.lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        elif value_lower in ("false", "0", "no", "off"):
            return False
        else:
            raise ValueError(f"Invalid boolean value for {key}: {value}")
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate batch size
        if self.BATCH_SIZE < 1:
            raise ValueError(f"BATCH_SIZE must be >= 1, got {self.BATCH_SIZE}")
        
        # Validate chunk sizes
        if self.CHUNK_SIZE < self.MIN_CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_SIZE ({self.CHUNK_SIZE}) must be >= MIN_CHUNK_SIZE "
                f"({self.MIN_CHUNK_SIZE})"
            )
        
        if self.CHUNK_OVERLAP < 0:
            raise ValueError(
                f"CHUNK_OVERLAP must be >= 0, got {self.CHUNK_OVERLAP}"
            )
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE "
                f"({self.CHUNK_SIZE})"
            )
        
        # Validate workers
        if self.MAX_WORKERS < 1:
            raise ValueError(f"MAX_WORKERS must be >= 1, got {self.MAX_WORKERS}")
        
        # Validate checkpoint interval
        if self.CHECKPOINT_INTERVAL < 1:
            raise ValueError(
                f"CHECKPOINT_INTERVAL must be >= 1, got {self.CHECKPOINT_INTERVAL}"
            )

    def _validate_model(self) -> None:
        """
        Validate model configuration against ModelRegistry.

        Logs warnings for unknown models but allows them to continue with generic adapter.
        """
        try:
            from ..embeddings.model_registry import get_model_registry
            registry = get_model_registry()

            # Check if model is registered
            if not registry.is_registered(self.MODEL_NAME):
                from ..utils.logger import get_default_logger
                logger = get_default_logger()
                logger.warning(
                    f"Model '{self.MODEL_NAME}' is not in the registered models list. "
                    "It will use a generic adapter fallback. "
                    f"Registered models: {registry.get_registered_models()}"
                )
            else:
                # Validate model name format (basic check)
                if not self.MODEL_NAME or not isinstance(self.MODEL_NAME, str):
                    raise ValueError(f"Invalid MODEL_NAME: {self.MODEL_NAME}. Must be a non-empty string.")

        except ImportError as e:
            # Graceful degradation if model registry is not available
            from ..utils.logger import get_default_logger
            logger = get_default_logger()
            logger.debug(f"Model registry not available for validation: {e}")
        except Exception as e:
            # Log validation errors but don't fail initialization
            from ..utils.logger import get_default_logger
            logger = get_default_logger()
            logger.warning(f"Model validation failed: {e}. Using model as-is.")

    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the configured model from ModelRegistry.

        Returns:
            Model metadata dictionary or None if not available
        """
        try:
            from ..embeddings.model_registry import get_model_registry
            registry = get_model_registry()
            return registry.get_model_metadata(self.MODEL_NAME)
        except Exception:
            return None

    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for the configured model.

        Returns:
            Embedding dimension (default: 1024 for backward compatibility)
        """
        metadata = self.get_model_metadata()
        if metadata:
            return metadata.embedding_dimension

        # Fallback for backward compatibility
        return 1024
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {
            "MODEL_NAME": self.MODEL_NAME,
            "MODEL_CACHE_DIR": self.MODEL_CACHE_DIR,
            "BATCH_SIZE": self.BATCH_SIZE,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "MIN_CHUNK_SIZE": self.MIN_CHUNK_SIZE,
            "VECTOR_DB_PATH": self.VECTOR_DB_PATH,
            "COLLECTION_NAME": self.COLLECTION_NAME,
            "DEVICE": self.DEVICE,
            "CUDA_VISIBLE_DEVICES": self.CUDA_VISIBLE_DEVICES,
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_FILE": self.LOG_FILE,
            "MAX_WORKERS": self.MAX_WORKERS,
            "ENABLE_CHECKPOINTING": self.ENABLE_CHECKPOINTING,
            "CHECKPOINT_INTERVAL": self.CHECKPOINT_INTERVAL,
        }

        # Add model metadata if available
        metadata = self.get_model_metadata()
        if metadata:
            config_dict["MODEL_METADATA"] = {
                "embedding_dimension": metadata.embedding_dimension,
                "max_sequence_length": metadata.max_sequence_length,
                "precision_requirements": metadata.precision_requirements,
                "adapter_class": metadata.adapter_class.__name__,
            }

        return config_dict
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        dim = self.get_embedding_dimension()
        return f"Config(device={self.DEVICE}, model={self.MODEL_NAME}, dim={dim}, batch_size={self.BATCH_SIZE})"


# Global configuration instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        env_file: Path to .env file. Only used on first call.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
