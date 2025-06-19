#!/usr/bin/env python3
"""
Stop script for the Adaptive Fake News Detector API
"""
import os
import sys
import signal
import subprocess
import time
from pathlib import Path

def find_server_process():
    """Find the running server process"""
    try:
        # Get the port from environment or use default
        port = os.getenv("PORT", "8000")
        
        # Find process using the port
        if sys.platform == "win32":
            # Windows command
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse the output to get PID
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            return int(parts[-1])
        else:
            # Unix/Linux command
            cmd = f"lsof -ti:{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
                
    except Exception as e:
        print(f"❌ Error finding server process: {e}")
    
    return None

def stop_server():
    """Stop the running server"""
    print("🛑 Stopping Adaptive Fake News Detector API...")
    print("=" * 50)
    
    # Find the server process
    pid = find_server_process()
    
    if pid is None:
        print("❌ No server process found running on port 8000")
        print("💡 Make sure the server is running before trying to stop it")
        return False
    
    print(f"📡 Found server process (PID: {pid})")
    
    try:
        if sys.platform == "win32":
            # Windows: Send SIGTERM equivalent
            subprocess.run(f"taskkill /PID {pid} /F", shell=True, check=True)
        else:
            # Unix/Linux: Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            
            # Wait a bit for graceful shutdown
            time.sleep(2)
            
            # If still running, force kill
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already terminated
        
        print("✅ Server stopped successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error stopping server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to stop the server"""
    print("🚀 Adaptive Fake News Detector API - Stop Script")
    print("=" * 60)
    
    # Check if server is running
    if not find_server_process():
        print("❌ No server found running on port 8000")
        print("💡 Start the server first using: python start_server.py")
        sys.exit(1)
    
    # Stop the server
    if stop_server():
        print("\n👋 Server has been stopped gracefully")
        print("💡 To restart, run: python start_server.py")
    else:
        print("\n❌ Failed to stop server")
        sys.exit(1)

if __name__ == "__main__":
    main() 