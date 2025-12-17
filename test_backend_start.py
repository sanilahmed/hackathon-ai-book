#!/usr/bin/env python3
"""
Simple test script to verify that the backend can start properly
"""
import subprocess
import time
import requests
import sys
import os

def test_backend_start():
    print("Testing backend startup...")

    # Change to backend directory
    os.chdir("backend")

    # Try to start the backend server in the background
    print("Starting backend server...")
    process = subprocess.Popen([
        sys.executable, "-c",
        """
import sys
sys.path.insert(0, '.')
try:
    from rag_agent_api.main import app
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='error')
except Exception as e:
    print(f'Error starting server: {e}')
    sys.exit(1)
        """
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Give the server some time to start
    time.sleep(5)

    # Check if the process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"Server failed to start. Stdout: {stdout.decode()}")
        print(f"Stderr: {stderr.decode()}")
        return False

    print("Backend server appears to be running!")

    # Try to make a health check request
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health check passed - backend is responding")
            health_data = response.json()
            print(f"Health status: {health_data.get('status', 'unknown')}")
            services = health_data.get('services', {})
            for service, status in services.items():
                print(f"  {service}: {status}")
        else:
            print(f"✗ Health check failed with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to backend server")
    except Exception as e:
        print(f"✗ Error during health check: {e}")

    # Try to get root endpoint
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✓ Root endpoint accessible")
        else:
            print(f"✗ Root endpoint returned status {response.status_code}")
    except:
        print("✗ Could not access root endpoint")

    # Terminate the process
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    print("Backend test completed.")
    return True

if __name__ == "__main__":
    test_backend_start()