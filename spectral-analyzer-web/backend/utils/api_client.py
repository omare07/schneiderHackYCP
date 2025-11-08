"""
API client for OpenRouter and other AI service integrations.

Provides async HTTP client with retry logic, rate limiting,
and comprehensive error handling for AI API interactions.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

import httpx
from httpx import AsyncClient, Response, RequestError, HTTPStatusError

from config.api_config import APIConfig, APIProvider
# Remove circular import - ConfigManager will be passed as parameter


@dataclass
class APIUsageStats:
    """API usage statistics tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    last_request_time: Optional[datetime] = None
    daily_requests: int = 0
    daily_cost: float = 0.0
    monthly_requests: int = 0
    monthly_cost: float = 0.0


@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            # Record this request
            self.requests.append(now)


class APIClient:
    """
    Async API client for AI services with comprehensive features.
    
    Features:
    - Async HTTP requests with retry logic
    - Rate limiting and cost tracking
    - Multiple provider support
    - Request/response logging
    - Error handling and fallbacks
    """
    
    def __init__(self, config_manager: Optional[Any] = None):
        """
        Initialize the API client.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.api_config = APIConfig()
        
        # HTTP client
        self.client: Optional[AsyncClient] = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        
        # Usage tracking
        self.usage_stats = APIUsageStats()
        
        # Cost tracking integration
        self.cost_tracker = None
        self._init_cost_tracker()
        
        # Request logging
        self.log_requests = False
        self.request_log_dir = None
        
        self.logger.debug("API client initialized")
    
    def _init_cost_tracker(self):
        """Initialize cost tracker if available."""
        try:
            from utils.cost_tracker import CostTracker
            self.cost_tracker = CostTracker(config_manager=self.config_manager)
        except Exception as e:
            self.logger.warning(f"Failed to initialize cost tracker: {e}")
            self.cost_tracker = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self.client is None:
            timeout = httpx.Timeout(30.0, connect=10.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            
            self.client = AsyncClient(
                timeout=timeout,
                limits=limits,
                follow_redirects=True
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def chat_completion(self, request: Dict[str, Any], 
                            provider: Optional[str] = None) -> APIResponse:
        """
        Send chat completion request to AI service.
        
        Args:
            request: Chat completion request payload
            provider: Optional provider override
            
        Returns:
            APIResponse with completion data
        """
        try:
            # Determine provider and model
            if provider is None:
                provider = self.api_config.current_provider.value
            
            model = request.get('model', self.api_config.current_model)
            
            # Get API configuration
            provider_enum = APIProvider(provider)
            endpoint_config = self.api_config.get_endpoint_config(provider_enum)
            
            if not endpoint_config:
                return APIResponse(
                    success=False,
                    error=f"No configuration found for provider: {provider}"
                )
            
            # Get API key
            api_key = self._get_api_key(provider)
            if not api_key:
                return APIResponse(
                    success=False,
                    error=f"No API key found for provider: {provider}"
                )
            
            # Prepare request
            url = f"{endpoint_config.base_url}{endpoint_config.chat_completion}"
            headers = endpoint_config.headers.copy()
            headers["Authorization"] = f"Bearer {api_key}"
            
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Make request with retry logic
            response = await self._make_request_with_retry(
                method="POST",
                url=url,
                headers=headers,
                json=request,
                max_retries=endpoint_config.max_retries,
                provider=provider
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Chat completion request failed: {e}")
            return APIResponse(
                success=False,
                error=str(e)
            )
    
    async def _make_request_with_retry(self, method: str, url: str,
                                     headers: Dict[str, str], json: Dict[str, Any],
                                     max_retries: int = 3, provider: str = 'unknown') -> APIResponse:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            json: Request payload
            max_retries: Maximum retry attempts
            
        Returns:
            APIResponse with request results
        """
        await self._ensure_client()
        
        last_error = None
        start_time = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                # Log request if enabled
                if self.log_requests:
                    self._log_request(method, url, headers, json, attempt)
                
                # Make request
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json
                )
                
                response_time = time.time() - start_time
                
                # Update usage stats
                self._update_usage_stats(response, response_time)
                
                # Log response if enabled
                if self.log_requests:
                    self._log_response(response, response_time)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Calculate tokens and cost
                tokens_used = self._extract_token_usage(response_data)
                cost = self._calculate_cost(json.get('model'), tokens_used)
                
                # Track cost if tracker available
                if self.cost_tracker and tokens_used and cost:
                    self.cost_tracker.track_api_call(
                        model=json.get('model', 'unknown'),
                        provider=provider,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time=response_time,
                        success=True,
                        cache_hit=False,
                        operation_type="api_call"
                    )
                
                return APIResponse(
                    success=True,
                    data=response_data,
                    status_code=response.status_code,
                    response_time=response_time,
                    tokens_used=tokens_used,
                    cost=cost
                )
                
            except HTTPStatusError as e:
                last_error = e
                
                # Handle specific HTTP errors
                if e.response.status_code == 429:  # Rate limited
                    wait_time = self._get_retry_wait_time(attempt, e.response)
                    self.logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                    
                elif e.response.status_code in [500, 502, 503, 504]:  # Server errors
                    if attempt < max_retries:
                        wait_time = self._get_retry_wait_time(attempt)
                        self.logger.warning(f"Server error, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Non-retryable error
                break
                
            except RequestError as e:
                last_error = e
                
                if attempt < max_retries:
                    wait_time = self._get_retry_wait_time(attempt)
                    self.logger.warning(f"Request error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                
                break
                
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error in request: {e}")
                break
        
        # All retries failed
        response_time = time.time() - start_time
        self.usage_stats.failed_requests += 1
        
        # Track failed request cost
        if self.cost_tracker:
            self.cost_tracker.track_api_call(
                model=json.get('model', 'unknown'),
                provider=provider,
                tokens_used=0,
                cost=0.0,
                response_time=response_time,
                success=False,
                cache_hit=False,
                operation_type="api_call",
                error_message=str(last_error)
            )
        
        return APIResponse(
            success=False,
            error=str(last_error),
            status_code=getattr(getattr(last_error, 'response', None), 'status_code', None),
            response_time=response_time
        )
    
    def _get_retry_wait_time(self, attempt: int, response: Optional[Response] = None) -> float:
        """
        Calculate wait time for retry attempts.
        
        Args:
            attempt: Current attempt number
            response: Optional HTTP response for rate limit headers
            
        Returns:
            Wait time in seconds
        """
        if response and response.status_code == 429:
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
            
            # Check for rate limit reset headers
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                try:
                    reset_timestamp = int(reset_time)
                    wait_time = reset_timestamp - int(time.time())
                    return max(1, wait_time)
                except ValueError:
                    pass
        
        # Exponential backoff
        return min(60, (2 ** attempt) + (time.time() % 1))
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider."""
        if self.config_manager:
            return self.config_manager.get_api_key(provider)
        return None
    
    def _extract_token_usage(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Extract token usage from API response."""
        try:
            usage = response_data.get('usage', {})
            return usage.get('total_tokens', 0)
        except Exception:
            return None
    
    def _calculate_cost(self, model: Optional[str], tokens: Optional[int]) -> Optional[float]:
        """Calculate request cost based on model and token usage."""
        if not model or not tokens:
            return None
        
        try:
            model_config = self.api_config.get_model_config(model)
            if model_config:
                return tokens * model_config.cost_per_token
        except Exception:
            pass
        
        return None
    
    def _update_usage_stats(self, response: Response, response_time: float):
        """Update usage statistics."""
        self.usage_stats.total_requests += 1
        self.usage_stats.last_request_time = datetime.now()
        
        if response.is_success:
            self.usage_stats.successful_requests += 1
        else:
            self.usage_stats.failed_requests += 1
    
    def _log_request(self, method: str, url: str, headers: Dict[str, str], 
                    json_data: Dict[str, Any], attempt: int):
        """Log API request for debugging."""
        try:
            # Sanitize headers (remove API key)
            safe_headers = headers.copy()
            if 'Authorization' in safe_headers:
                safe_headers['Authorization'] = 'Bearer [REDACTED]'
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'url': url,
                'headers': safe_headers,
                'payload': json_data,
                'attempt': attempt
            }
            
            self.logger.debug(f"API Request: {json.dumps(log_entry, indent=2)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log request: {e}")
    
    def _log_response(self, response: Response, response_time: float):
        """Log API response for debugging."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'status_code': response.status_code,
                'response_time': response_time,
                'headers': dict(response.headers),
                'content_length': len(response.content) if response.content else 0
            }
            
            # Include response body for errors or if specifically enabled
            if not response.is_success:
                try:
                    log_entry['response_body'] = response.json()
                except Exception:
                    log_entry['response_body'] = response.text[:1000]  # Truncate
            
            self.logger.debug(f"API Response: {json.dumps(log_entry, indent=2)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log response: {e}")
    
    async def test_connection(self, provider: str, model: str) -> APIResponse:
        """
        Test API connection with a simple request.
        
        Args:
            provider: API provider to test
            model: Model to test with
            
        Returns:
            APIResponse with test results
        """
        try:
            # Simple test request
            test_request = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a connection test. Please respond with 'OK'."
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = await self.chat_completion(test_request, provider)
            
            if response.success:
                # Check if response contains expected content
                try:
                    content = response.data['choices'][0]['message']['content']
                    if 'OK' in content.upper():
                        return APIResponse(
                            success=True,
                            data={'message': 'Connection test successful'},
                            response_time=response.response_time
                        )
                    else:
                        return APIResponse(
                            success=True,
                            data={'message': 'Connection successful but unexpected response'},
                            response_time=response.response_time
                        )
                except Exception as e:
                    return APIResponse(
                        success=False,
                        error=f"Failed to parse test response: {e}"
                    )
            else:
                return response
                
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Connection test failed: {e}"
            )
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics including cost tracking data.
        
        Returns:
            Dictionary with comprehensive usage statistics
        """
        base_stats = {
            'total_requests': self.usage_stats.total_requests,
            'successful_requests': self.usage_stats.successful_requests,
            'failed_requests': self.usage_stats.failed_requests,
            'success_rate': (
                self.usage_stats.successful_requests / max(1, self.usage_stats.total_requests)
            ) * 100,
            'total_tokens': self.usage_stats.total_tokens,
            'total_cost': self.usage_stats.total_cost,
            'last_request_time': (
                self.usage_stats.last_request_time.isoformat()
                if self.usage_stats.last_request_time else None
            ),
            'daily_requests': self.usage_stats.daily_requests,
            'daily_cost': self.usage_stats.daily_cost,
            'monthly_requests': self.usage_stats.monthly_requests,
            'monthly_cost': self.usage_stats.monthly_cost
        }
        
        # Add cost tracker statistics if available
        if self.cost_tracker:
            try:
                cost_stats = self.cost_tracker.get_usage_statistics()
                base_stats['cost_tracking'] = cost_stats.to_dict()
            except Exception as e:
                self.logger.warning(f"Failed to get cost tracking stats: {e}")
        
        return base_stats
    
    def enable_request_logging(self, log_dir: Optional[str] = None):
        """
        Enable request/response logging for debugging.
        
        Args:
            log_dir: Optional directory for log files
        """
        self.log_requests = True
        self.request_log_dir = log_dir
        self.logger.info("API request logging enabled")
    
    def disable_request_logging(self):
        """Disable request/response logging."""
        self.log_requests = False
        self.request_log_dir = None
        self.logger.info("API request logging disabled")
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = APIUsageStats()
        self.logger.info("Usage statistics reset")


class OpenRouterClient:
    """
    Specialized client for OpenRouter API with CSV normalization support.
    
    Features:
    - Optimized for x-ai/grok-4-fast model
    - CSV structure analysis prompts
    - Cost tracking and optimization
    - Structured JSON response handling
    """
    
    def __init__(self, api_key: str, config_manager: Optional[Any] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.config_manager = config_manager
        
        # Initialize base API client
        self.api_client = APIClient(config_manager)
        
        # OpenRouter specific configuration
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = "x-ai/grok-4-fast"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://spectral-analyzer.mrglabs.com",
            "X-Title": "MRG Labs Spectral Analyzer"
        }
        
        # Usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'normalization_requests': 0,
            'cache_hits': 0
        }
        
        # Cost tracking integration
        self.cost_tracker = None
        self._init_cost_tracker()
        
        self.logger.debug("OpenRouter client initialized")
    
    def _init_cost_tracker(self):
        """Initialize cost tracker if available."""
        try:
            from utils.cost_tracker import CostTracker
            self.cost_tracker = CostTracker(config_manager=self.config_manager)
        except Exception as e:
            self.logger.warning(f"Failed to initialize cost tracker in OpenRouter client: {e}")
            self.cost_tracker = None
    
    async def analyze_csv_structure(self, csv_preview: str, file_info: Dict[str, Any],
                                  model: str = None) -> Dict[str, Any]:
        """
        Analyze CSV structure using AI for normalization.
        
        Args:
            csv_preview: Preview of CSV data (first 50-100 rows)
            file_info: File metadata and information
            model: Optional model override
            
        Returns:
            Dictionary with AI analysis results
        """
        try:
            if model is None:
                model = self.default_model
            
            # Create structured prompt for CSV analysis
            prompt = self._create_csv_analysis_prompt(csv_preview, file_info)
            
            # Prepare request
            request = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert spectroscopy data analyst. Analyze CSV files and provide structured normalization plans in valid JSON format. Be conservative with confidence scores - only use high confidence (>0.9) when very certain about column mappings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 2000,
                "response_format": {"type": "json_object"}
            }
            
            # Make API request
            response = await self._make_request(request)
            
            if response.success:
                self.usage_stats['normalization_requests'] += 1
                return response.data
            else:
                raise Exception(f"API request failed: {response.error}")
                
        except Exception as e:
            self.logger.error(f"CSV structure analysis failed: {e}")
            self.usage_stats['failed_requests'] += 1
            raise
    
    def _create_csv_analysis_prompt(self, csv_preview: str, file_info: Dict[str, Any]) -> str:
        """Create structured prompt for CSV analysis."""
        return f"""
Analyze this spectroscopy CSV data and create a normalization plan to convert it to standard format.

EXPECTED STANDARD FORMAT:
- Column 1: Wavenumber (cm⁻¹), range 400-4000, typically in descending order
- Column 2: Absorbance, Transmittance, or Intensity values
- Additional columns: Sample metadata (optional)

CSV DATA PREVIEW:
{csv_preview}

FILE INFORMATION:
{json.dumps(file_info, indent=2)}

ANALYSIS REQUIREMENTS:
1. Identify wavenumber/frequency columns (look for cm⁻¹, wavenumber, frequency, wave number patterns)
2. Identify intensity columns (absorbance, transmittance, intensity, signal patterns)
3. Detect data quality issues and anomalies
4. Recommend necessary transformations
5. Provide confidence scores for each mapping decision (0.0-1.0)

RESPONSE FORMAT (JSON):
{{
    "can_normalize": true/false,
    "confidence": 0.0-1.0,
    "detected_format": {{
        "delimiter": ",",
        "decimal_separator": ".",
        "has_headers": true/false,
        "metadata_rows": 0,
        "encoding": "utf-8"
    }},
    "column_mapping": {{
        "wavenumber_column": "column_name_or_null",
        "absorbance_column": "column_name_or_null",
        "transmittance_column": "column_name_or_null",
        "intensity_column": "column_name_or_null"
    }},
    "column_mappings": [
        {{
            "original_name": "original_column_name",
            "target_name": "wavenumber|absorbance|transmittance|intensity|metadata",
            "data_type": "numeric|text|categorical",
            "transformation": "none|unit_conversion|scale_factor|reverse_order|other",
            "confidence": 0.0-1.0,
            "notes": "explanation of mapping decision"
        }}
    ],
    "transformations": [
        {{
            "type": "skip_rows|rename_columns|reverse_order|convert_units|scale_values|interpolate_missing",
            "parameters": {{}},
            "reason": "explanation"
        }}
    ],
    "warnings": ["list of potential issues"],
    "recommendations": ["list of recommendations"],
    "confidence_score": 0-100,
    "analysis_notes": "overall analysis summary"
}}

CONFIDENCE GUIDELINES:
- High (>90): Very clear column patterns, standard format, no ambiguity
- Medium (70-90): Reasonable certainty but some ambiguity or non-standard format
- Low (<70): Unclear patterns, multiple interpretations possible, or significant issues

Focus on accuracy and be conservative with confidence scores.
"""
    
    async def _make_request(self, request: Dict[str, Any]) -> APIResponse:
        """
        Make request to OpenRouter API with error handling.
        
        Args:
            request: API request payload
            
        Returns:
            APIResponse with results
        """
        try:
            start_time = time.time()
            
            # Use the base API client for the actual request
            response = await self.api_client.chat_completion(request, "openrouter")
            
            # Track usage
            self.usage_stats['total_requests'] += 1
            
            if response.success:
                self.usage_stats['successful_requests'] += 1
                
                # Extract and parse JSON response
                if response.data and 'choices' in response.data:
                    content = response.data['choices'][0]['message']['content']
                    try:
                        parsed_data = json.loads(content)
                        
                        # Track tokens and cost
                        if response.tokens_used:
                            self.usage_stats['total_tokens'] += response.tokens_used
                        if response.cost:
                            self.usage_stats['total_cost'] += response.cost
                        
                        # Track with cost tracker if available
                        if self.cost_tracker and response.tokens_used and response.cost:
                            self.cost_tracker.track_api_call(
                                model=request.get('model', self.default_model),
                                provider="openrouter",
                                tokens_used=response.tokens_used,
                                cost=response.cost,
                                response_time=response.response_time,
                                success=True,
                                cache_hit=False,
                                operation_type="csv_analysis"
                            )
                        
                        return APIResponse(
                            success=True,
                            data=parsed_data,
                            status_code=response.status_code,
                            response_time=response.response_time,
                            tokens_used=response.tokens_used,
                            cost=response.cost
                        )
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON response: {e}")
                        return APIResponse(
                            success=False,
                            error=f"Invalid JSON response: {e}",
                            response_time=time.time() - start_time
                        )
                else:
                    return APIResponse(
                        success=False,
                        error="Invalid API response format",
                        response_time=time.time() - start_time
                    )
            else:
                self.usage_stats['failed_requests'] += 1
                return response
                
        except Exception as e:
            self.usage_stats['failed_requests'] += 1
            self.logger.error(f"OpenRouter API request failed: {e}")
            return APIResponse(
                success=False,
                error=str(e),
                response_time=time.time() - start_time if 'start_time' in locals() else None
            )
    
    async def test_connection(self, model: str = None) -> APIResponse:
        """
        Test OpenRouter API connection.
        
        Args:
            model: Optional model to test with
            
        Returns:
            APIResponse with test results
        """
        try:
            if model is None:
                model = self.default_model
            
            test_request = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a connection test. Please respond with 'OK' in JSON format: {\"status\": \"OK\"}"
                    }
                ],
                "max_tokens": 50,
                "temperature": 0,
                "response_format": {"type": "json_object"}
            }
            
            response = await self._make_request(test_request)
            
            if response.success:
                try:
                    if response.data.get('status') == 'OK':
                        return APIResponse(
                            success=True,
                            data={'message': 'OpenRouter connection test successful'},
                            response_time=response.response_time
                        )
                    else:
                        return APIResponse(
                            success=True,
                            data={'message': 'Connection successful but unexpected response'},
                            response_time=response.response_time
                        )
                except Exception:
                    return APIResponse(
                        success=True,
                        data={'message': 'Connection successful'},
                        response_time=response.response_time
                    )
            else:
                return response
                
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Connection test failed: {e}"
            )
    
    def track_usage(self, tokens_used: int, cost: float):
        """
        Track API usage and costs.
        
        Args:
            tokens_used: Number of tokens consumed
            cost: Cost of the request
        """
        self.usage_stats['total_tokens'] += tokens_used
        self.usage_stats['total_cost'] += cost
    
    def track_cache_hit(self):
        """Track cache hit for cost optimization metrics."""
        self.usage_stats['cache_hits'] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        total_requests = self.usage_stats['total_requests']
        cache_hit_rate = 0.0
        
        if total_requests > 0:
            cache_hit_rate = (self.usage_stats['cache_hits'] / total_requests) * 100
        
        return {
            **self.usage_stats,
            'cache_hit_rate': cache_hit_rate,
            'success_rate': (
                self.usage_stats['successful_requests'] / max(1, total_requests)
            ) * 100,
            'average_cost_per_request': (
                self.usage_stats['total_cost'] / max(1, self.usage_stats['successful_requests'])
            )
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'normalization_requests': 0,
            'cache_hits': 0
        }
        self.logger.info("OpenRouter usage statistics reset")
    
    async def close(self):
        """Close the client and cleanup resources."""
        await self.api_client.close()