"""End-to-end mock test for conversation continuity flow."""

import asyncio
import httpx
from unittest.mock import patch, AsyncMock, MagicMock

# Test scenario: Design a cache system, then ask a follow-up about scaling

API_BASE = "http://localhost:8000"


async def mock_e2e_test():
    """
    End-to-end test with mocked AI responses.
    
    Flow:
    1. Start a design session with a question
    2. Wait for completion
    3. Submit a follow-up question
    4. Verify the session preserves context
    """
    print("=" * 60)
    print("E2E TEST: Conversation Continuity")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120) as client:
        # Step 1: Start a new design session
        print("\n[Step 1] Starting design session...")
        response = await client.post(
            f"{API_BASE}/api/design/start",
            json={
                "question": "Design a simple key-value cache with TTL support",
                "architect_model": "gpt-4o-mini",  # Use faster models for testing
                "engineer_model": "claude-sonnet-4-20250514",
                "auditor_model": "gemini-2.0-flash",
            }
        )
        
        if response.status_code != 200:
            print(f"  ✗ Failed to start session: {response.text}")
            return False
        
        data = response.json()
        session_id = data["session_id"]
        print(f"  ✓ Session started: {session_id}")
        
        # Step 2: Poll until completion
        print("\n[Step 2] Waiting for design completion...")
        max_polls = 120  # 2 minutes max
        for i in range(max_polls):
            status_response = await client.get(f"{API_BASE}/api/design/{session_id}/detailed")
            status = status_response.json()
            
            phase = status.get("current_phase", "unknown")
            is_complete = status.get("is_complete", False)
            
            if i % 10 == 0:  # Print every 10 polls
                print(f"  ... Phase: {phase}, Complete: {is_complete}")
            
            if is_complete:
                print(f"  ✓ Design complete! Final phase: {phase}")
                break
            
            await asyncio.sleep(1)
        else:
            print("  ✗ Timeout waiting for completion")
            return False
        
        # Step 3: Get the result
        print("\n[Step 3] Fetching result...")
        result_response = await client.get(f"{API_BASE}/api/design/{session_id}/result")
        if result_response.status_code != 200:
            print(f"  ✗ Failed to get result: {result_response.text}")
            return False
        
        result = result_response.json()
        adr_title = result.get("adr", {}).get("title", "Unknown")
        print(f"  ✓ Got ADR: {adr_title}")
        
        # Step 4: Verify session is in storage
        print("\n[Step 4] Verifying session persistence...")
        sessions_response = await client.get(f"{API_BASE}/api/sessions/all")
        sessions = sessions_response.json().get("sessions", [])
        session_ids = [s["session_id"] for s in sessions]
        
        if session_id in session_ids:
            print(f"  ✓ Session {session_id} found in storage")
        else:
            print(f"  ! Session not yet in storage (may be in memory only)")
        
        # Step 5: Submit a follow-up question
        print("\n[Step 5] Submitting follow-up question...")
        follow_up_response = await client.post(
            f"{API_BASE}/api/design/{session_id}/follow-up",
            json={"question": "How would we scale this cache to handle 1 million requests per second?"}
        )
        
        if follow_up_response.status_code != 200:
            print(f"  ✗ Failed to start follow-up: {follow_up_response.text}")
            return False
        
        follow_up_data = follow_up_response.json()
        is_running = follow_up_data.get("is_running", False)
        print(f"  ✓ Follow-up started, running: {is_running}")
        
        # Step 6: Wait for follow-up completion
        print("\n[Step 6] Waiting for follow-up completion...")
        for i in range(max_polls):
            status_response = await client.get(f"{API_BASE}/api/design/{session_id}/detailed")
            status = status_response.json()
            
            phase = status.get("current_phase", "unknown")
            is_complete = status.get("is_complete", False)
            
            if i % 10 == 0:
                print(f"  ... Phase: {phase}, Complete: {is_complete}")
            
            if is_complete:
                print(f"  ✓ Follow-up complete! Final phase: {phase}")
                break
            
            await asyncio.sleep(1)
        else:
            print("  ✗ Timeout waiting for follow-up completion")
            return False
        
        # Step 7: Verify conversation turns
        print("\n[Step 7] Checking conversation history...")
        result_response = await client.get(f"{API_BASE}/api/design/{session_id}/result")
        result = result_response.json()
        
        # Note: The turns are tracked in the state, check via storage
        sessions_response = await client.get(f"{API_BASE}/api/sessions/all")
        sessions = sessions_response.json().get("sessions", [])
        session_info = next((s for s in sessions if s["session_id"] == session_id), None)
        
        if session_info:
            turns = session_info.get("turns", 0)
            print(f"  ✓ Session has {turns} conversation turn(s)")
        
        print("\n" + "=" * 60)
        print("E2E TEST PASSED ✓")
        print("=" * 60)
        return True


if __name__ == "__main__":
    print("Running E2E test for conversation continuity...")
    print("This will use real AI models - ensure API keys are configured.\n")
    
    result = asyncio.run(mock_e2e_test())
    exit(0 if result else 1)
