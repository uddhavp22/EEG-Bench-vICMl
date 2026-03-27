#!/bin/bash

SRC="/radraid2/spanchavati/EEGBench/data/make_dataset/"
DEST="./data/make_dataset/"

# Create destination if it doesn't exist
mkdir -p "$DEST"

echo "Scanning for largest files per task..."

# 1. Get all unique 'prefixes' (everything before the final _NUMBER.h5)
# We look for files containing 'LaBraMModel', extract the prefix, and get unique list.
PREFIXES=$(ls "$SRC" | grep "LaBraMModel" | sed -E 's/_[0-9]+\.h5$//' | sort | uniq)

for PREFIX in $PREFIXES; do
    # 2. For this specific prefix, find the file with the largest version number
    # We list files starting with the prefix, sort by version (natural numbers), take the last one.
    LARGEST_FILE=$(ls "$SRC" | grep "^$PREFIX" | sort -V | tail -n 1)
    
    if [ ! -z "$LARGEST_FILE" ]; then
        echo "Found largest for $PREFIX -> $LARGEST_FILE"
        
        # 3. Dry run rsync for this specific file
        rsync -av "$SRC$LARGEST_FILE" "$DEST" #n
    fi
done