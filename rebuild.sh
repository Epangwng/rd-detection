#!/bin/bash
set -e

echo "=== 1. Stopping & Removing Old Containers ==="
sudo docker stop rd-detection-app 2>/dev/null || true
sudo docker rm rd-detection-app 2>/dev/null || true

echo "=== 2. Cleaning Old Docker Images and Build Cache ==="
sudo docker rmi -f rd-app 2>/dev/null || true
sudo docker builder prune -f

echo "=== 3. Building Fresh Docker Image (No Cache) ==="
sudo docker build --no-cache -t rd-app .

echo "=== 4. Starting New Container on Port 5000 ==="
sudo docker run -d -p 5000:7860 --restart always --name rd-detection-app rd-app

echo "=== 5. Deployment Complete! Streaming Logs (Ctrl+C to exit) ==="
sudo docker logs -f rd-detection-app
