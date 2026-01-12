
import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000/api"

def test_full_flow():
    print("🚀 Starting End-to-End Backend Test")
    
    # 1. Start Design
    print("\n1. Starting Design Session...")
    payload = {"question": "Design a simple notification system"}
    start_resp = requests.post(f"{BASE_URL}/design/start", json=payload)
    if start_resp.status_code != 200:
        print(f"❌ Failed to start: {start_resp.text}")
        return
    
    data = start_resp.json()
    session_id = data["session_id"]
    print(f"✅ Session Started: {session_id}")
    
    # 2. Poll for Completion (Ideation -> Cross Critique -> Audit -> Convergence)
    print("\n2. Polling for Completion...")
    for i in range(300):
        status_resp = requests.get(f"{BASE_URL}/design/{session_id}/detailed")
        status = status_resp.json()

        phase = status.get("current_phase")
        completed = status.get("is_complete")

        sys.stdout.write(f"\rStatus: {phase}, Complete: {completed} ({i}s)   ")
        sys.stdout.flush()

        if completed:
            print(f"\n✅ Session Completed Successfully!")
            print(f"Final Phase: {phase}")

            # 3. Get Result
            result_resp = requests.get(f"{BASE_URL}/design/{session_id}/result")
            if result_resp.status_code == 200:
                print("✅ Result retrieved successfully")
                print(json.dumps(result_resp.json(), indent=2))
            else:
                print(f"❌ Failed to get result: {result_resp.text}")
            return

        time.sleep(1)
    else:
        print("\n❌ Timed out waiting for completion")

if __name__ == "__main__":
    test_full_flow()
