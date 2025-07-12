import subprocess
import json
import time
from datetime import datetime
import argparse
import os

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
    parser = argparse.ArgumentParser(description="Log GPU usage every 5 seconds to a JSON Lines file.")
    parser.add_argument("label", type=str, help="Label to identify the log file (e.g., 'experiment1', 'runA')")
    args = parser.parse_args()

    log_filename = f"gpu_log_{args.label}.jsonl"
    print(f"Logging GPU info every 5 seconds...")
    print(f"Writing to {log_filename}")
    print("Press Ctrl+C to stop.")

    with open(log_filename, 'a') as log_file:
        while True:
            metrics = get_gpu_metrics()
            for entry in metrics:
                line = json.dumps(entry)
                print(line)
                log_file.write(line + '\n')
                log_file.flush()
            time.sleep(5)

if __name__ == "__main__":
    main()
