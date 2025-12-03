"""
LLM Factory Module

Creates and manages LLM instances for different agents based on configuration.
Supports multiple providers: OpenAI, Anthropic, Groq, Azure OpenAI, and Ollama.
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from src.utils.config import Config


class LLMFactory:
    """Factory for creating LLM instances for different agents."""
    
    _config: Optional[Config] = None
    _llm_cache: Dict[str, Any] = {}
    
    @classmethod
    def initialize(cls, config: Optional[Config] = None) -> None:
        """Initialize the factory with configuration."""
        cls._config = config or Config()
        cls._llm_cache = {}
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure factory is initialized."""
        if cls._config is None:
            cls.initialize()
    
    @classmethod
    def create_llm(
        cls,
        provider: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> Any:
        """
        Create an LLM instance based on provider and model.
        
        Args:
            provider: LLM provider (openai, anthropic, groq, azure, ollama)
            model: Model name/identifier
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens for response
            api_key: API key for the provider (uses config if not provided)
            cache_key: Key for caching the LLM instance
            
        Returns:
            LLM instance configured for the specified provider
        """
        cls._ensure_initialized()
        
        # Check cache first
        if cache_key and cache_key in cls._llm_cache:
            return cls._llm_cache[cache_key]
        
        provider = provider.lower().strip()
        
        if provider == "openai":
            llm = cls._create_openai_llm(model, temperature, max_tokens, api_key)
        elif provider == "anthropic":
            llm = cls._create_anthropic_llm(model, temperature, max_tokens, api_key)
        elif provider == "groq":
            llm = cls._create_groq_llm(model, temperature, max_tokens, api_key)
        elif provider == "azure":
            llm = cls._create_azure_llm(model, temperature, max_tokens, api_key)
        elif provider == "ollama":
            llm = cls._create_ollama_llm(model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Cache the LLM if cache key is provided
        if cache_key:
            cls._llm_cache[cache_key] = llm
        
        return llm
    
    @classmethod
    def _create_openai_llm(
        cls,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str]
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        api_key = api_key or (cls._config.OPENAI_API_KEY if cls._config else None)
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    
    @classmethod
    def _create_anthropic_llm(
        cls,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str]
    ) -> Any:
        """Create Anthropic LLM instance."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Install it with: pip install langchain-anthropic"
            )
        
        api_key = api_key or (cls._config.ANTHROPIC_API_KEY if cls._config else None)
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in configuration")
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
    
    @classmethod
    def _create_groq_llm(
        cls,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str]
    ) -> Any:
        """Create Groq LLM instance."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq is not installed. "
                "Install it with: pip install langchain-groq"
            )
        
        api_key = api_key or (cls._config.GROQ_API_KEY if cls._config else None)
        
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration")
        
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
    
    @classmethod
    def _create_azure_llm(
        cls,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str]
    ) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance."""
        api_key = api_key or (cls._config.AZURE_OPENAI_API_KEY if cls._config else None)
        endpoint = cls._config.AZURE_OPENAI_ENDPOINT if cls._config else None
        api_version = cls._config.AZURE_OPENAI_API_VERSION if cls._config else "2024-02-15-preview"
        
        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required")
        
        return AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            model=model,
            temperature=temperature
        )
    
    @classmethod
    def _create_ollama_llm(
        cls,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Any:
        """Create Ollama local LLM instance."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )
        
        base_url = cls._config.OLLAMA_BASE_URL if cls._config else "http://localhost:11434"
        ollama_model = model or (cls._config.OLLAMA_MODEL if cls._config else "mistral")
        
        return ChatOllama(
            model=ollama_model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
    
    @classmethod
    def create_query_analyzer_llm(cls) -> Any:
        """Create LLM for Query Analyzer agent."""
        cls._ensure_initialized()
        assert cls._config is not None
        
        return cls.create_llm(
            provider=cls._config.AI_QUERY_ANALYZER_PROVIDER,
            model=cls._config.AI_QUERY_ANALYZER_MODEL,
            temperature=cls._config.AI_QUERY_ANALYZER_TEMPERATURE,
            max_tokens=cls._config.AI_QUERY_ANALYZER_MAX_TOKENS,
            api_key=cls._config.AI_QUERY_ANALYZER_API_KEY,
            cache_key="query_analyzer"
        )
    
    @classmethod
    def create_clarification_llm(cls) -> Any:
        """Create LLM for Clarification agent."""
        cls._ensure_initialized()
        assert cls._config is not None
        
        return cls.create_llm(
            provider=cls._config.AI_CLARIFICATION_PROVIDER,
            model=cls._config.AI_CLARIFICATION_MODEL,
            temperature=cls._config.AI_CLARIFICATION_TEMPERATURE,
            max_tokens=cls._config.AI_CLARIFICATION_MAX_TOKENS,
            api_key=cls._config.AI_CLARIFICATION_API_KEY,
            cache_key="clarification"
        )
    
    @classmethod
    def create_query_rewriter_llm(cls) -> Any:
        """Create LLM for Query Rewriter agent."""
        cls._ensure_initialized()
        assert cls._config is not None
        
        return cls.create_llm(
            provider=cls._config.AI_QUERY_REWRITER_PROVIDER,
            model=cls._config.AI_QUERY_REWRITER_MODEL,
            temperature=cls._config.AI_QUERY_REWRITER_TEMPERATURE,
            max_tokens=cls._config.AI_QUERY_REWRITER_MAX_TOKENS,
            api_key=cls._config.AI_QUERY_REWRITER_API_KEY,
            cache_key="query_rewriter"
        )
    
    @classmethod
    def create_rag_generator_llm(cls) -> Any:
        """Create LLM for RAG Response Generator agent."""
        cls._ensure_initialized()
        assert cls._config is not None
        
        return cls.create_llm(
            provider=cls._config.AI_RAG_GENERATOR_PROVIDER,
            model=cls._config.AI_RAG_GENERATOR_MODEL,
            temperature=cls._config.AI_RAG_GENERATOR_TEMPERATURE,
            max_tokens=cls._config.AI_RAG_GENERATOR_MAX_TOKENS,
            api_key=cls._config.AI_RAG_GENERATOR_API_KEY,
            cache_key="rag_generator"
        )
