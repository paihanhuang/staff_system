
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.adapters.openai_adapter import ArchitectAdapter
from src.adapters.anthropic_adapter import EngineerAdapter
from src.adapters.google_adapter import AuditorAdapter

async def test_openai_connectivity():
    """Test actual connectivity to OpenAI API."""
    print("\nTesting OpenAI Connectivity...")
    try:
        adapter = ArchitectAdapter()
        # Simple generation
        response = await adapter.generate("Hello, return the word 'Working'.")
        print(f"✅ OpenAI Response: {response}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

async def test_anthropic_connectivity():
    """Test actual connectivity to Anthropic API."""
    print("\nTesting Anthropic Connectivity...")
    try:
        adapter = EngineerAdapter()
        # Simple generation
        response = await adapter.generate("Hello, return the word 'Working'.")
        print(f"✅ Anthropic Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Anthropic API failed: {e}")
        return False

async def test_google_connectivity():
    """Test actual connectivity to Google API."""
    print("\nTesting Google Connectivity...")
    try:
        adapter = AuditorAdapter()
        # Simple generation
        response = await adapter.generate("Hello, return the word 'Working'.")
        print(f"✅ Google Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Google API failed: {e}")
        return False

async def main():
    print("Starting API Connectivity Tests...")
    
    openai_ok = await test_openai_connectivity()
    anthropic_ok = await test_anthropic_connectivity()
    google_ok = await test_google_connectivity()
    
    print("\nSummary:")
    print(f"OpenAI: {'✅ OK' if openai_ok else '❌ FAILED'}")
    print(f"Anthropic: {'✅ OK' if anthropic_ok else '❌ FAILED'}")
    print(f"Google: {'✅ OK' if google_ok else '❌ FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())
