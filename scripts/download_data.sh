#!/bin/bash

# Setting strict mode
set -euo pipefail

# Defining the directories
DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
LOG_DIR="logs"
DOWNLOAD_URL="https://example.com/dataset.zip"  # Replace with actual URL

# Creating directories if they don't exist
mkdir -p $DATA_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR

# To log file for download script
LOG_FILE="$LOG_DIR/download_data.log"

# To start logging
exec > >(tee -i $LOG_FILE)
exec 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting data download."

# To download the dataset
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Downloading dataset from $DOWNLOAD_URL."
curl -o "$DATA_DIR/dataset.zip" -L $DOWNLOAD_URL

# To check if download was successful
if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to download dataset from $DOWNLOAD_URL."
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Download completed."

# To unzip the dataset
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Unzipping dataset."
unzip -o "$DATA_DIR/dataset.zip" -d $DATA_DIR

# Checking if unzip was successful
if [ $? -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to unzip dataset."
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Unzip completed."

# Removing the zip file to save space
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Removing zip file."
rm "$DATA_DIR/dataset.zip"

#  To log the contents of the data directory
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Data directory contents:"
ls -lh $DATA_DIR

# To end logging
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Data download and extraction process completed successfully."
