#!/usr/bin/env bash

# Usage: ./sample_pdfs.sh <number_of_pdfs_to_sample> <source_folder> <destination_folder>
# Example: ./sample_pdfs.sh 10 papers test_sample

N=${1:-10}  # Number of files to sample, default=10 if not provided
SRC=${2:-papers}  # Source folder
DEST=${3:-test_sample}  # Destination folder

# Create destination folder if it doesn't exist
mkdir -p "$DEST"

# Shuffle and sample N pdfs from source, copy to destination
find "$SRC" -maxdepth 1 -type f -iname "*.pdf" | shuf -n "$N" | while read -r file; do
    echo "Copying: $file"
    cp "$file" "$DEST"/
done

echo "Done. Sampled $N PDFs into $DEST/"
