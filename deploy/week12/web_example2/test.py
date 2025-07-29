#!/usr/bin/env python3
"""
Test script for the Gene Logo Generator Flask application
"""

import requests
import time

def test_flask_app():
    """Test the Flask application endpoints"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("Testing Gene Logo Generator Flask App")
    print("=" * 50)
    
    # Test 1: Check if the server is running
    print("1. Testing server connection...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✓ Server is running successfully")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server: {e}")
        print("Make sure you're running 'python site.py' in another terminal")
        return
    
    # Test 2: Test home page with gene parameter
    print("\n2. Testing home page with gene parameter...")
    try:
        response = requests.get(f"{base_url}/?name=BRCA1", timeout=10)
        if response.status_code == 200:
            print("✓ Home page with gene parameter works")
        else:
            print(f"✗ Home page test failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Home page test error: {e}")
    
    # Test 3: Test logo generation for common genes
    test_genes = ["BRCA1", "TP53", "EGFR"]
    
    print(f"\n3. Testing logo generation for genes: {', '.join(test_genes)}")
    
    for gene in test_genes:
        print(f"\n   Testing gene: {gene}")
        try:
            url = f"{base_url}/logo/{gene}.png"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Check if it's actually a PNG image
                if response.headers.get('content-type') == 'image/png':
                    print(f"   ✓ {gene} logo generated successfully (PNG, {len(response.content)} bytes)")
                    
                    # Optionally save the image for manual inspection
                    with open(f"test_{gene}_logo.png", "wb") as f:
                        f.write(response.content)
                    print(f"   → Saved as test_{gene}_logo.png")
                else:
                    print(f"   ✗ {gene} returned non-PNG content: {response.headers.get('content-type')}")
            elif response.status_code == 404:
                print(f"   ⚠ {gene} not found in database (404)")
            elif response.status_code == 500:
                print(f"   ✗ {gene} caused server error (500) - check logs")
            else:
                print(f"   ✗ {gene} returned status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"   ✗ {gene} request timed out (>30s)")
        except requests.exceptions.RequestException as e:
            print(f"   ✗ {gene} request error: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"\n{'='*50}")
    print("Test completed!")
    print("\nTo manually test:")
    print(f"1. Open browser to: {base_url}")
    print("2. Enter a gene name and click 'Generate Logo'")
    print("3. Check generated PNG files in current directory")

if __name__ == "__main__":
    test_flask_app()