#!/bin/bash
# Watch training progress - regenerates the plot every 5 seconds
LOG="${1:-training_log.csv}"
echo "Watching $LOG for changes (Ctrl+C to stop)"
while true; do
    if [ -f "$LOG" ]; then
        python3 plot_training.sh "$LOG" 2>/dev/null
    fi
    sleep 5
done
