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
            available_servers[s['id']] = {
                'host': s['host'],
                'country': s['country'],
                '_latency': None,  # Initialize latency to None
                '_load': None  # Initialize load to None
            }

    # Measure latency for each server
    for server_id, server in available_servers.items():
        try:
            st.get_servers([server_id])
            st.get_best_server()
            available_servers[server_id]['_latency'] = st.results.ping
        except Exception as e:
            print(f"Error measuring latency for server {server_id}: {e}")

    return available_servers


def check_internet_speed(available_servers, server_id=None):
    try:
        st = speedtest.Speedtest()
        if server_id is not None:
            # Convert server_id to string for consistent comparison
            server_id = str(server_id)
        
        best_server = None
        if server_id and server_id in available_servers:
            best_server = available_servers[server_id]
            print(f"Using server: {best_server['host']} in {best_server['country']}")
        else:
            if server_id:
                print(f"Server ID {server_id} not found. Using best available server.")
            st.get_best_server()
            best_server = st.get_best_server()
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
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    available_servers = get_available_servers()
    for server_id, server in available_servers.items():
        print(f"Server ID: {server_id}")
        print(f"Host: {server['host']}")
        print(f"Country: {server['country']}")
        print(f"Latency: {server['_latency']} ms")
        print(f"Load: {server['_load']}")
        print("----")
    speed_results = check_internet_speed(available_servers, server_id=32643)
    if speed_results:
        print(f"\nRaw Results (Mbps)")
        print(json.dumps(speed_results, indent=2, ensure_ascii=False))
    else:
        print("Failed to retrieve speed test results.")
