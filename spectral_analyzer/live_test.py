#!/usr/bin/env python3
"""
Live test of AI normalization engine with real OpenRouter API.

This script demonstrates the complete AI normalization workflow
using the actual OpenRouter API with Grok-4-Fast model.
"""

import asyncio
import pandas as pd
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the spectral_analyzer directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import OpenRouterClient

# Get API key from environment
API_KEY = os.getenv('OPENROUTER_API_KEY')
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


async def test_openrouter_connection():
    """Test OpenRouter API connection with real API key."""
    print("Testing OpenRouter API connection...")
    
    api_key = API_KEY
    
    try:
        client = OpenRouterClient(api_key)
        
        # Test connection
        response = await client.test_connection()
        
        if response.success:
            print("✓ OpenRouter API connection successful!")
            print(f"  Response: {response.data.get('message', 'Connected')}")
            print(f"  Response time: {response.response_time:.2f}s")
            return True
        else:
            print(f"✗ OpenRouter API connection failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ OpenRouter API test failed: {e}")
        return False


async def test_csv_analysis():
    """Test CSV analysis with real AI."""
    print("\nTesting CSV analysis with AI...")
    
    api_key = API_KEY
    
    try:
        client = OpenRouterClient(api_key)
        
        # Create sample problematic CSV data
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
            "column_names": ["Wave Number (cm-1)", "Transmittance %", "Sample ID"],
            "dtypes": {"Wave Number (cm-1)": "int64", "Transmittance %": "int64", "Sample ID": "object"},
            "sample_values": {
                "Wave Number (cm-1)": [500, 1000, 1500],
                "Transmittance %": [90, 80, 70],
                "Sample ID": ["A1", "A1", "A1"]
            }
        }
        
        print("Sending CSV data to AI for analysis...")
        print("CSV Preview:")
        print(csv_preview)
        print("\nFile Info:")
        print(json.dumps(file_info, indent=2))
        
        # Analyze CSV structure
        response = await client.analyze_csv_structure(csv_preview, file_info)
        
        if response.success:
            print("\n✓ AI analysis successful!")
            print("AI Response:")
            print(json.dumps(response.data, indent=2))
            
            # Check if AI can normalize
            if response.data.get('can_normalize'):
                print(f"\n✓ AI confidence: {response.data.get('confidence', 0):.2f}")
                print(f"✓ Confidence score: {response.data.get('confidence_score', 0)}")
                
                # Show column mappings
                mappings = response.data.get('column_mappings', [])
                print(f"\n✓ Found {len(mappings)} column mappings:")
                for mapping in mappings:
                    print(f"  - {mapping['original_name']} → {mapping['target_name']} (confidence: {mapping['confidence']:.2f})")
                
                # Show transformations
                transformations = response.data.get('transformations', [])
                print(f"\n✓ Recommended {len(transformations)} transformations:")
                for transform in transformations:
                    print(f"  - {transform['type']}: {transform['reason']}")
                
                # Show warnings and recommendations
                warnings = response.data.get('warnings', [])
                if warnings:
                    print(f"\n⚠️  Warnings ({len(warnings)}):")
                    for warning in warnings:
                        print(f"  - {warning}")
                
                recommendations = response.data.get('recommendations', [])
                if recommendations:
                    print(f"\n💡 Recommendations ({len(recommendations)}):")
                    for rec in recommendations:
                        print(f"  - {rec}")
                
            else:
                print("✗ AI determined this CSV cannot be normalized")
            
            return True
        else:
            print(f"✗ AI analysis failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ CSV analysis test failed: {e}")
        return False


async def test_usage_tracking():
    """Test usage tracking and cost monitoring."""
    print("\nTesting usage tracking...")
    
    api_key = API_KEY
    
    try:
        client = OpenRouterClient(api_key)
        
        # Track some usage
        client.track_usage(150, 0.0015)
        client.track_cache_hit()
        
        stats = client.get_usage_stats()
        
        print("✓ Usage statistics:")
        print(f"  - Total requests: {stats['total_requests']}")
        print(f"  - Successful requests: {stats['successful_requests']}")
        print(f"  - Failed requests: {stats['failed_requests']}")
        print(f"  - Total tokens: {stats['total_tokens']}")
        print(f"  - Total cost: ${stats['total_cost']:.6f}")
        print(f"  - Cache hits: {stats['cache_hits']}")
        print(f"  - Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"  - Success rate: {stats['success_rate']:.1f}%")
        print(f"  - Average cost per request: ${stats['average_cost_per_request']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Usage tracking test failed: {e}")
        return False


async def test_different_csv_formats():
    """Test AI with different CSV formats."""
    print("\nTesting different CSV formats...")
    
    api_key = API_KEY
    
    test_cases = [
        {
            "name": "Standard Format",
            "csv": "Wavenumber,Absorbance\n4000,0.1\n3000,0.2\n2000,0.3\n1000,0.4",
            "expected_confidence": "high"
        },
        {
            "name": "European Format",
            "csv": "Wellenzahl;Absorption\n4000;0,1\n3000;0,2\n2000;0,3\n1000;0,4",
            "expected_confidence": "medium"
        },
        {
            "name": "Messy Headers",
            "csv": "Wave # (cm-1),Abs. Value,Notes\n4000,0.1,good\n3000,0.2,ok\n2000,0.3,fair\n1000,0.4,poor",
            "expected_confidence": "medium"
        }
    ]
    
    try:
        client = OpenRouterClient(api_key)
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n  Test {i+1}: {test_case['name']}")
            print(f"  CSV: {test_case['csv'][:50]}...")
            
            file_info = {
                "path": f"test_{i+1}.csv",
                "rows": 4,
                "columns": test_case['csv'].split('\n')[0].count(',') + 1 if ',' in test_case['csv'] else test_case['csv'].split('\n')[0].count(';') + 1,
                "column_names": test_case['csv'].split('\n')[0].split(',' if ',' in test_case['csv'] else ';')
            }
            
            try:
                response = await client.analyze_csv_structure(test_case['csv'], file_info)
                
                if response.success:
                    confidence = response.data.get('confidence_score', 0)
                    can_normalize = response.data.get('can_normalize', False)
                    
                    print(f"    ✓ Can normalize: {can_normalize}")
                    print(f"    ✓ Confidence: {confidence}%")
                    
                    results.append({
                        'name': test_case['name'],
                        'success': True,
                        'confidence': confidence,
                        'can_normalize': can_normalize
                    })
                else:
                    print(f"    ✗ Analysis failed: {response.error}")
                    results.append({
                        'name': test_case['name'],
                        'success': False,
                        'error': response.error
                    })
                    
            except Exception as e:
                print(f"    ✗ Test failed: {e}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n✓ Tested {len(test_cases)} different CSV formats")
        successful = sum(1 for r in results if r['success'])
        print(f"✓ {successful}/{len(test_cases)} tests successful")
        
        return successful == len(test_cases)
        
    except Exception as e:
        print(f"✗ Different CSV formats test failed: {e}")
        return False


async def run_live_tests():
    """Run all live tests with real OpenRouter API."""
    print("=" * 70)
    print("AI NORMALIZATION ENGINE - LIVE TESTS WITH OPENROUTER API")
    print("=" * 70)
    print("Using Grok-4-Fast model for intelligent CSV normalization")
    
    tests = [
        ("OpenRouter Connection", test_openrouter_connection()),
        ("CSV Analysis with AI", test_csv_analysis()),
        ("Usage Tracking", test_usage_tracking()),
        ("Different CSV Formats", test_different_csv_formats()),
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
    print(f"LIVE TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 All live tests passed! AI normalization engine is working perfectly!")
        print("\n🚀 Key Features Demonstrated:")
        print("   ✓ OpenRouter API integration with Grok-4-Fast")
        print("   ✓ Intelligent CSV structure analysis")
        print("   ✓ Confidence-based decision making")
        print("   ✓ Column mapping with confidence scores")
        print("   ✓ Transformation recommendations")
        print("   ✓ Usage tracking and cost monitoring")
        print("   ✓ Multiple CSV format support")
        print("   ✓ Error handling and fallback behavior")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_live_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)