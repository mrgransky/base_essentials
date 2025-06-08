import speedtest
from tabulate import tabulate  # For formatted output
import time
import json



def get_available_servers():
		st = speedtest.Speedtest()
		servers = st.get_servers()
		available_servers = {}
		for server in servers.values():
				for s in server:
						available_servers[s['id']] = s
		for server_id, server in available_servers.items():
				print(f"ID: {server_id}, Host: {server['host']}, Country: {server['country']}")

def check_internet_speed(server_id=None):
    try:
        st = speedtest.Speedtest()
        servers = st.get_servers()
        available_servers = {}
        for server in servers.values():
            for s in server:
                available_servers[s['id']] = s
        
        if server_id:
            if server_id in available_servers:
                best_server = available_servers[server_id]
            else:
                print(f"Server ID {server_id} not found. Using best available server.")
                st.get_best_server()
                best_server = st.get_best_server()
        else:
            st.get_best_server()
            best_server = st.get_best_server()
        
        if best_server:
            print(f"Using server: {best_server['host']} in {best_server['country']}")
            
            print("Testing download speed...")
            download_speed = st.download()
            
            print("Testing upload speed...")
            upload_speed = st.upload()
            
            ping = st.results.ping
            
            return {
                "download": download_speed / 1e6,
                "upload": upload_speed / 1e6,
                "ping": ping,
            }
        else:
            print("Failed to retrieve speed test results.")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
		get_available_servers()
		speed_results = check_internet_speed(server_id=13712)
		if speed_results:
				print(f"\nRaw Results (Mbps)")
				print(json.dumps(speed_results, indent=2, ensure_ascii=False))
		else:
				print("Failed to retrieve speed test results.")
