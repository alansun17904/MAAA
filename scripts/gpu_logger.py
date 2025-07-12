import subprocess
import time
import json
from datetime import datetime

LOG_FILE = "gpu_log.jsonl"  # Each line is a JSON object

def log_nvidia_smi():
    while True:
        timestamp = datetime.now().isoformat()
        try:
            output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            output_text = output.decode("utf-8")
        except subprocess.CalledProcessError as e:
            output_text = f"Error: {e.output.decode('utf-8')}"

        log_entry = {
            "timestamp": timestamp,
            "nvidia_smi_output": output_text
        }

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        time.sleep(5)

if __name__ == "__main__":
    log_nvidia_smi()
