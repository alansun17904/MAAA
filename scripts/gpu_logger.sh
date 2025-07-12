#!/bin/bash

LOG_FILE="gpu_log.jsonl"

while true; do
    TIMESTAMP=$(date -Iseconds)
    OUTPUT=$(nvidia-smi)

    {
        echo "{"
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"nvidia_smi_output\": $(echo "$OUTPUT" | jq -Rs .)"
        echo "}"
    } >> "$LOG_FILE"

    sleep 5
done
