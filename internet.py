import speedtest
from tabulate import tabulate  # For formatted output
import time
import json

def check_internet_speed():
		"""
		Measures internet speed (download, upload, and ping) using speedtest-cli.
		Returns results in a formatted dictionary and prints a summary.
		"""
		try:
				# Initialize speed test
				st = speedtest.Speedtest()
				print("Searching for best server...")
				best_server = st.get_best_server()
				print(f"Found server: {best_server['host']} in {best_server['country']}")

				# Perform tests
				print("Testing download speed...")
				download_speed = st.download()
				time.sleep(2)
				print("Testing upload speed...")
				upload_speed = st.upload()
				ping = st.results.ping

				# Convert to Mbps and format
				results = {
						"Metric": ["Download Speed", "Upload Speed", "Ping"],
						"Value": [
								f"{download_speed / 1e6:.2f} Mbps",
								f"{upload_speed / 1e6:.2f} Mbps",
								f"{ping:.2f} ms"
						]
				}

				# Print table
				print("\nInternet Speed Test Results:")
				print(tabulate(zip(results["Metric"], results["Value"]), headers=["Metric", "Value"], tablefmt="grid"))
				
				return {
						"download": download_speed / 1e6,
						"upload": upload_speed / 1e6,
						"ping": ping
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
		speed_results = check_internet_speed()
		if speed_results:
				print(f"\nRaw Results (Mbps)")
				print(json.dumps(speed_results, indent=2, ensure_ascii=False))
