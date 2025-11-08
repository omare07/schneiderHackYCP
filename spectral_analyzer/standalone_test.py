#!/usr/bin/env python3
"""
Standalone test of OpenRouter API integration.

This script tests the OpenRouter client directly without project dependencies.
"""

import asyncio
import json
import sys
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx


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


class StandaloneOpenRouterClient:
    """
    Standalone OpenRouter client for testing AI normalization.
    """
    
    def __init__(self, api_key: str):
        """Initialize OpenRouter client."""
        self.api_key = api_key
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
            'total_cost': 0.0
        }
    
    async def test_connection(self) -> APIResponse:
        """Test OpenRouter API connection."""
        try:
            test_request = {
                "model": self.default_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a connection test. Please respond with 'OK'."
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = await self._make_request(test_request)
            return response
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Connection test failed: {e}"
            )
    
    async def analyze_csv_structure(self, csv_preview: str, file_info: Dict[str, Any]) -> APIResponse:
        """Analyze CSV structure using AI."""
        try:
            prompt = self._create_csv_analysis_prompt(csv_preview, file_info)
            
            request = {
                "model": self.default_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert spectroscopy data analyst. Analyze CSV files and provide structured normalization plans in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = await self._make_request(request)
            
            if response.success and response.data:
                # Try to parse the content as JSON
                try:
                    content = response.data['choices'][0]['message']['content']
                    # Clean up the content to extract JSON
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    parsed_data = json.loads(content)
                    return APIResponse(
                        success=True,
                        data=parsed_data,
                        response_time=response.response_time,
                        tokens_used=response.tokens_used
                    )
                except json.JSONDecodeError as e:
                    return APIResponse(
                        success=False,
                        error=f"Failed to parse AI response as JSON: {e}",
                        data={'raw_content': content if 'content' in locals() else 'No content'}
                    )
            
            return response
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"CSV analysis failed: {e}"
            )
    
    def _create_csv_analysis_prompt(self, csv_preview: str, file_info: Dict[str, Any]) -> str:
        """Create structured prompt for CSV analysis."""
        return f"""
Analyze this spectroscopy CSV data and create a normalization plan.

EXPECTED STANDARD FORMAT:
- Column 1: Wavenumber (cm⁻¹), range 400-4000, typically descending order
- Column 2: Absorbance, Transmittance, or Intensity values
- Additional columns: Sample metadata (optional)

CSV DATA PREVIEW:
{csv_preview}

FILE INFORMATION:
{json.dumps(file_info, indent=2)}

Please provide a JSON response with this exact structure:
{{
    "can_normalize": true/false,
    "confidence": 0.0-1.0,
    "confidence_score": 0-100,
    "detected_format": {{
        "delimiter": ",",
        "decimal_separator": ".",
        "has_headers": true/false,
        "metadata_rows": 0,
        "encoding": "utf-8"
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
            "type": "skip_rows|rename_columns|reverse_order|convert_units|scale_values",
            "parameters": {{}},
            "reason": "explanation"
        }}
    ],
    "warnings": ["list of potential issues"],
    "recommendations": ["list of recommendations"],
    "analysis_notes": "overall analysis summary"
}}

Focus on accuracy and be conservative with confidence scores.
"""
    
    async def _make_request(self, request: Dict[str, Any]) -> APIResponse:
        """Make request to OpenRouter API."""
        start_time = time.time()
        
        try:
            self.usage_stats['total_requests'] += 1
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=request
                )
                
                response_time = time.time() - start_time
                
                if response.is_success:
                    self.usage_stats['successful_requests'] += 1
                    data = response.json()
                    
                    # Extract token usage
                    tokens_used = None
                    if 'usage' in data:
                        tokens_used = data['usage'].get('total_tokens', 0)
                        self.usage_stats['total_tokens'] += tokens_used
                    
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code,
                        response_time=response_time,
                        tokens_used=tokens_used
                    )
                else:
                    self.usage_stats['failed_requests'] += 1
                    error_text = response.text
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status_code}: {error_text}",
                        status_code=response.status_code,
                        response_time=response_time
                    )
                    
        except Exception as e:
            self.usage_stats['failed_requests'] += 1
            return APIResponse(
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = self.usage_stats['total_requests']
        return {
            **self.usage_stats,
            'success_rate': (self.usage_stats['successful_requests'] / max(1, total)) * 100
        }


async def test_openrouter_connection():
    """Test OpenRouter API connection."""
    print("Testing OpenRouter API connection...")
    
    api_key = "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"
    
    try:
        client = StandaloneOpenRouterClient(api_key)
        response = await client.test_connection()
        
        if response.success:
            print("✓ OpenRouter API connection successful!")
            print(f"  Response time: {response.response_time:.2f}s")
            if response.data and 'choices' in response.data:
                content = response.data['choices'][0]['message']['content']
                print(f"  AI Response: {content}")
            return True
        else:
            print(f"✗ OpenRouter API connection failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


async def test_csv_analysis():
    """Test CSV analysis with real AI."""
    print("\nTesting CSV analysis with AI...")
    
    api_key = "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"
    
    try:
        client = StandaloneOpenRouterClient(api_key)
        
        # Test case: Problematic CSV that needs normalization
        csv_preview = """Wave Number (cm-1),Transmittance %,Sample ID
500,90,A1
1000,80,A1
1500,70,A1
2000,60,A1
2500,50,A1
3000,40,A1
3500,30,A1
4000,20,A1"""
        
        file_info = {
            "path": "test_problematic.csv",
            "rows": 8,
            "columns": 3,
            "column_names": ["Wave Number (cm-1)", "Transmittance %", "Sample ID"]
        }
        
        print("Sending CSV to AI for analysis...")
        print("CSV Preview:")
        print(csv_preview[:100] + "...")
        
        response = await client.analyze_csv_structure(csv_preview, file_info)
        
        if response.success:
            print("\n✓ AI analysis successful!")
            
            data = response.data
            print(f"✓ Can normalize: {data.get('can_normalize', False)}")
            print(f"✓ Confidence: {data.get('confidence', 0):.2f}")
            print(f"✓ Confidence score: {data.get('confidence_score', 0)}")
            
            # Show column mappings
            mappings = data.get('column_mappings', [])
            print(f"\n✓ Column mappings ({len(mappings)}):")
            for mapping in mappings:
                print(f"  - {mapping['original_name']} → {mapping['target_name']} (confidence: {mapping['confidence']:.2f})")
            
            # Show transformations
            transformations = data.get('transformations', [])
            print(f"\n✓ Transformations ({len(transformations)}):")
            for transform in transformations:
                print(f"  - {transform['type']}: {transform['reason']}")
            
            # Show warnings
            warnings = data.get('warnings', [])
            if warnings:
                print(f"\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            # Show recommendations
            recommendations = data.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
            
            return True
        else:
            print(f"✗ AI analysis failed: {response.error}")
            if response.data and 'raw_content' in response.data:
                print(f"Raw AI response: {response.data['raw_content'][:200]}...")
            return False
            
    except Exception as e:
        print(f"✗ CSV analysis test failed: {e}")
        return False


async def test_usage_tracking():
    """Test usage tracking."""
    print("\nTesting usage tracking...")
    
    api_key = "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"
    
    try:
        client = StandaloneOpenRouterClient(api_key)
        
        # Make a simple request to generate usage
        await client.test_connection()
        
        stats = client.get_usage_stats()
        
        print("✓ Usage statistics:")
        print(f"  - Total requests: {stats['total_requests']}")
        print(f"  - Successful requests: {stats['successful_requests']}")
        print(f"  - Failed requests: {stats['failed_requests']}")
        print(f"  - Success rate: {stats['success_rate']:.1f}%")
        print(f"  - Total tokens: {stats['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Usage tracking test failed: {e}")
        return False


async def run_standalone_tests():
    """Run standalone tests."""
    print("=" * 70)
    print("AI NORMALIZATION ENGINE - STANDALONE OPENROUTER TESTS")
    print("=" * 70)
    print("Testing OpenRouter API integration with Grok-4-Fast model")
    
    tests = [
        ("OpenRouter Connection", test_openrouter_connection()),
        ("CSV Analysis with AI", test_csv_analysis()),
        ("Usage Tracking", test_usage_tracking()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = await test_coro
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"STANDALONE TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 All standalone tests passed!")
        print("\n🚀 OpenRouter Integration Verified:")
        print("   ✓ API connection and authentication")
        print("   ✓ AI-powered CSV structure analysis")
        print("   ✓ Intelligent column mapping")
        print("   ✓ Transformation recommendations")
        print("   ✓ Usage tracking and monitoring")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_standalone_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)