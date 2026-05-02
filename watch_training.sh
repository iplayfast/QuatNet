#!/bin/bash
# Watch training progress: refresh plot every 5 seconds
LOG="${1:-training_log.csv}"
echo "Watching $LOG (Ctrl+C to stop)"
while true; do
    if [ -f "$LOG" ]; then
        python3 plot_training.sh "$LOG" 2>/dev/null
        xdg-open images/training_log.png 2>/dev/null &
        IMG_PID=$!
        sleep 5
        kill $IMG_PID 2>/dev/null
    else
        sleep 5
    fi
done
