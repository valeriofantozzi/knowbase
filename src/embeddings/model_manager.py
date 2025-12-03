"""
Model Manager Module

Manages multiple model instances with caching and memory management.
Replaces singleton pattern to enable switching between different models.
"""

from typing import Dict, List, Optional, Tuple
import time
from collections import OrderedDict

from .model_loader import ModelLoader
from ..utils.logger import get_default_logger


class ModelManager:
    """
    Manages multiple model instances with intelligent caching and memory management.

    This class replaces the singleton pattern to allow switching between different
    embedding models while maintaining efficient memory usage through caching.
    """

    def __init__(self, max_cache_size: int = 3):
        """
        Initialize model manager.

        Args:
            max_cache_size: Maximum number of models to keep in cache
        """
        self.max_cache_size = max_cache_size
        self.logger = get_default_logger()

        # Model cache: model_name -> (ModelLoader, last_access_time, memory_usage_estimate)
        self._model_cache: OrderedDict[str, Tuple[ModelLoader, float, int]] = OrderedDict()

        # Current active model
        self._current_model: Optional[str] = None

    def get_model_loader(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> ModelLoader:
        """
        Get or create a model loader for the specified model.

        Args:
            model_name: Name of the model to load
            device: Device to use for the model
            cache_dir: Cache directory for model files

        Returns:
            ModelLoader instance for the requested model
        """
        # Check if model is already cached
        if model_name in self._model_cache:
            loader, _, _ = self._model_cache[model_name]
            # Update access time
            self._model_cache.move_to_end(model_name)
            self._update_access_time(model_name)
            return loader

        # Model not in cache, create new one
        self.logger.info(f"Creating new model loader for: {model_name}")

        loader = ModelLoader(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir
        )

        # Estimate memory usage (rough approximation)
        memory_usage = self._estimate_memory_usage(loader)

        # Add to cache
        self._model_cache[model_name] = (loader, time.time(), memory_usage)

        # Set as current model
        self._current_model = model_name

        # Clean up cache if needed
        self._cleanup_cache()

        return loader

    def switch_model(self, model_name: str) -> ModelLoader:
        """
        Switch to a different model, unloading the current one if different.

        Args:
            model_name: Name of the model to switch to

        Returns:
            ModelLoader instance for the new model
        """
        if self._current_model == model_name:
            # Already using this model
            return self.get_model_loader(model_name)

        self.logger.info(f"Switching from {self._current_model} to {model_name}")

        # Get the new model (this will handle caching)
        new_loader = self.get_model_loader(model_name)

        # Update current model
        self._current_model = model_name

        return new_loader

    def unload_model(self, model_name: str) -> bool:
        """
        Explicitly unload a model from cache.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if model_name not in self._model_cache:
            self.logger.warning(f"Model {model_name} not found in cache")
            return False

        loader, _, _ = self._model_cache[model_name]

        # Unload the model
        loader.unload_model()

        # Remove from cache
        del self._model_cache[model_name]

        # Update current model if it was unloaded
        if self._current_model == model_name:
            self._current_model = None

        self.logger.info(f"Unloaded model: {model_name}")
        return True

    def list_loaded_models(self) -> List[Dict[str, any]]:
        """
        Get list of currently loaded models with their information.

        Returns:
            List of dictionaries containing model information
        """
        models_info = []

        for model_name, (loader, last_access, memory_usage) in self._model_cache.items():
            info = loader.get_model_info()
            info.update({
                "is_current": model_name == self._current_model,
                "last_access": last_access,
                "memory_usage_mb": memory_usage,
                "cache_position": list(self._model_cache.keys()).index(model_name)
            })
            models_info.append(info)

        return models_info

    def clear_cache(self) -> int:
        """
        Clear all models from cache.

        Returns:
            Number of models unloaded
        """
        unloaded_count = 0

        for model_name in list(self._model_cache.keys()):
            self.unload_model(model_name)
            unloaded_count += 1

        self._current_model = None
        self.logger.info(f"Cleared cache, unloaded {unloaded_count} models")
        return unloaded_count

    def unload_unused_models(self, max_models_to_keep: Optional[int] = None) -> int:
        """
        Unload least recently used models to stay within cache limits.

        Args:
            max_models_to_keep: Maximum number of models to keep (uses self.max_cache_size if None)

        Returns:
            Number of models unloaded
        """
        if max_models_to_keep is None:
            max_models_to_keep = self.max_cache_size

        if len(self._model_cache) <= max_models_to_keep:
            return 0

        # Sort by access time (oldest first)
        sorted_models = sorted(
            self._model_cache.items(),
            key=lambda x: x[1][1]  # Sort by access time
        )

        # Keep the most recently used models
        models_to_unload = sorted_models[:-max_models_to_keep]
        unloaded_count = 0

        for model_name, _ in models_to_unload:
            self.unload_model(model_name)
            unloaded_count += 1

        self.logger.info(f"Unloaded {unloaded_count} unused models")
        return unloaded_count

    def get_current_model(self) -> Optional[str]:
        """
        Get the name of the currently active model.

        Returns:
            Current model name or None if no model is active
        """
        return self._current_model

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_memory = sum(memory for _, _, memory in self._model_cache.values())

        return {
            "cached_models": len(self._model_cache),
            "max_cache_size": self.max_cache_size,
            "total_memory_usage_mb": total_memory,
            "current_model": self._current_model,
            "cache_utilization": len(self._model_cache) / self.max_cache_size if self.max_cache_size > 0 else 0
        }

    def _update_access_time(self, model_name: str) -> None:
        """Update the access time for a cached model."""
        if model_name in self._model_cache:
            loader, _, memory = self._model_cache[model_name]
            self._model_cache[model_name] = (loader, time.time(), memory)

    def _estimate_memory_usage(self, loader: ModelLoader) -> int:
        """
        Estimate memory usage of a model in MB.

        This is a rough approximation based on parameter count.
        """
        try:
            model_info = loader.get_model_info()
            total_params = model_info.get("total_parameters", 0)

            # Rough estimation: ~4 bytes per parameter for float32
            # Plus some overhead for model structure
            memory_bytes = total_params * 4 * 1.5  # 1.5x overhead
            memory_mb = memory_bytes / (1024 * 1024)

            return int(memory_mb)
        except Exception:
            # Fallback estimate
            return 1000  # 1GB default estimate

    def _cleanup_cache(self) -> None:
        """Clean up cache if it exceeds maximum size."""
        if len(self._model_cache) > self.max_cache_size:
            self.unload_unused_models(self.max_cache_size)


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get global model manager instance.

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
