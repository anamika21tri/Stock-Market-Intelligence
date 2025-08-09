"""
Rate limiting module for handling API rate limits and anti-bot measures
"""
import time
import asyncio
import random
from typing import Optional, Callable, Any
from dataclasses import dataclass
from threading import Lock
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..utils import get_logger, get_config

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    delay_between_requests: float = 1.5
    backoff_multiplier: float = 2.0
    max_backoff_time: float = 300.0
    jitter_range: tuple[float, float] = (0.1, 0.3)


class RateLimiter:
    """
    Advanced rate limiter with exponential backoff and jitter
    Handles both synchronous and asynchronous operations
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or self._load_config()
        self._request_times = deque()
        self._lock = Lock()
        self._backoff_time = self.config.delay_between_requests
        self._consecutive_failures = 0
        
        logger.info(f"Rate limiter initialized: {self.config.requests_per_minute} requests/min")
    
    def _load_config(self) -> RateLimitConfig:
        """Load rate limiting configuration"""
        try:
            config_data = get_config().data_collection.rate_limiting
            return RateLimitConfig(
                requests_per_minute=config_data.get('requests_per_minute', 60),
                delay_between_requests=config_data.get('delay_between_requests', 1.5),
                backoff_multiplier=config_data.get('backoff_multiplier', 2.0),
                max_backoff_time=config_data.get('max_backoff_time', 300.0),
                jitter_range=(0.1, 0.3)
            )
        except Exception as e:
            logger.warning(f"Failed to load rate limit config, using defaults: {e}")
            return RateLimitConfig()
    
    def _clean_old_requests(self) -> None:
        """Remove request times older than 1 minute"""
        cutoff_time = time.time() - 60
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()
    
    def _calculate_delay(self) -> float:
        """Calculate delay with jitter"""
        base_delay = self._backoff_time
        jitter_min, jitter_max = self.config.jitter_range
        jitter = random.uniform(jitter_min, jitter_max)
        return base_delay * (1 + jitter)
    
    def _update_backoff(self, success: bool) -> None:
        """Update backoff time based on success/failure"""
        if success:
            # Reset backoff on success
            self._consecutive_failures = 0
            self._backoff_time = self.config.delay_between_requests
        else:
            # Increase backoff on failure
            self._consecutive_failures += 1
            self._backoff_time = min(
                self._backoff_time * self.config.backoff_multiplier,
                self.config.max_backoff_time
            )
            logger.warning(f"Request failed, backoff increased to {self._backoff_time:.2f}s")
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        with self._lock:
            self._clean_old_requests()
            
            # Check if we're at the rate limit
            if len(self._request_times) >= self.config.requests_per_minute:
                # Calculate how long to wait
                oldest_request = self._request_times[0]
                wait_time = 60 - (time.time() - oldest_request)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self._clean_old_requests()
            
            # Add jitter delay
            delay = self._calculate_delay()
            time.sleep(delay)
            
            # Record this request
            self._request_times.append(time.time())
    
    async def async_wait_if_needed(self) -> None:
        """Async version of wait_if_needed"""
        with self._lock:
            self._clean_old_requests()
            
            if len(self._request_times) >= self.config.requests_per_minute:
                oldest_request = self._request_times[0]
                wait_time = 60 - (time.time() - oldest_request)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    self._clean_old_requests()
            
            delay = self._calculate_delay()
            await asyncio.sleep(delay)
            
            self._request_times.append(time.time())
    
    def execute_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limiting and backoff"""
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                self.wait_if_needed()
                result = func(*args, **kwargs)
                self._update_backoff(success=True)
                return result
                
            except Exception as e:
                self._update_backoff(success=False)
                
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for function {func.__name__}: {e}")
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                time.sleep(self._backoff_time)
    
    async def async_execute_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of execute_with_backoff"""
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                await self.async_wait_if_needed()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self._update_backoff(success=True)
                return result
                
            except Exception as e:
                self._update_backoff(success=False)
                
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for function {func.__name__}: {e}")
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                await asyncio.sleep(self._backoff_time)


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on response patterns
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config)
        self._success_rate_window = deque(maxlen=100)
        self._response_times = deque(maxlen=50)
        self._last_adaptation = time.time()
    
    def _adapt_rate_limit(self) -> None:
        """Adapt rate limiting based on success rate and response times"""
        now = time.time()
        
        # Only adapt every 30 seconds
        if now - self._last_adaptation < 30:
            return
        
        if len(self._success_rate_window) < 10:
            return
        
        success_rate = sum(self._success_rate_window) / len(self._success_rate_window)
        avg_response_time = sum(self._response_times) / len(self._response_times) if self._response_times else 1.0
        
        # Adjust rate based on success rate
        if success_rate > 0.9 and avg_response_time < 2.0:
            # High success rate and fast responses - can be more aggressive
            self.config.requests_per_minute = min(self.config.requests_per_minute + 5, 120)
            self.config.delay_between_requests = max(self.config.delay_between_requests * 0.9, 0.5)
        elif success_rate < 0.7 or avg_response_time > 5.0:
            # Low success rate or slow responses - be more conservative
            self.config.requests_per_minute = max(self.config.requests_per_minute - 5, 20)
            self.config.delay_between_requests = min(self.config.delay_between_requests * 1.2, 5.0)
        
        logger.info(f"Adapted rate limit: {self.config.requests_per_minute} req/min, "
                   f"delay: {self.config.delay_between_requests:.2f}s")
        
        self._last_adaptation = now
    
    def record_request_outcome(self, success: bool, response_time: float) -> None:
        """Record the outcome of a request for adaptive learning"""
        self._success_rate_window.append(1 if success else 0)
        self._response_times.append(response_time)
        self._adapt_rate_limit()


# Global rate limiter instance
_global_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter()
    return _global_rate_limiter
