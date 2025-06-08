import speedtest
from tabulate import tabulate  # For formatted output
import time
import json

def check_internet_speed(server_id=None):
    try:
        st = speedtest.Speedtest()
        if server_id:
            st.get_servers({server_id})
            best_server = st.get_best_server()
        else:
            best_server = st.get_best_server()
            
        print(f"Using server: {best_server['host']} in {best_server['country']}")
        
        print("Testing download speed...")
        download_speed = st.download()
        time.sleep(2)
        
        print("Testing upload speed...")
        upload_speed = st.upload()
        time.sleep(2)
        
        ping = st.results.ping
        
        return {
            "download": download_speed / 1e6,
            "upload": upload_speed / 1e6,
            "ping": ping,
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
		# Install speedtest-cli if not already installed
		try:
				import speedtest
		except ImportError:
				print("Installing speedtest-cli...")
				import subprocess
				subprocess.run(["pip", "install", "speedtest-cli"], check=True)
				import speedtest

		# Run speed test
		results = check_internet_speed(server_id=13712)
		if speed_results:
				print(f"\nRaw Results (Mbps)")
				print(json.dumps(speed_results, indent=2, ensure_ascii=False))
