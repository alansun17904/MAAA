import subprocess
import json
import time
from datetime import datetime

LOG_FILE = "gpu_log.jsonl"
INTERVAL = 5  # seconds

print(f"Logging GPU info every {INTERVAL} seconds...")
print(f"Writing to {LOG_FILE}")
print("Press Ctrl+C to stop.")

def get_gpu_metrics():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        metrics = []

        for line in lines:
            util, mem_used, mem_total, power = [x.strip() for x in line.split(',')]
            entry = {
                "timestamp": datetime.now().isoformat(),
                "gpu_utilization": int(util),
                "memory_used": int(mem_used),
                "memory_total": int(mem_total),
                "power_draw": float(power)
            }
            metrics.append(entry)

        return metrics

    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e)
        return []

def main():
    with open(LOG_FILE, 'a') as log_file:
        while True:
            metrics = get_gpu_metrics()
            for entry in metrics:
                line = json.dumps(entry)
                print(line)
                log_file.write(line + '\n')
                log_file.flush()
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
