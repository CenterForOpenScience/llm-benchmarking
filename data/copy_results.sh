#!/bin/bash

IDS=(2 4 7 20 21 22 23 24 26)

for id in "${IDS[@]}"; do
    echo "----------------------------------------"
    echo "Processing ID: $id"

    SRC_INPUT="original/$id/input"
    SRC_EVALS="original/$id/evals"
    DEST_DIR="results/native/$id/o3"

    # Check if source input exists to avoid errors on bad IDs
    if [ ! -d "$SRC_INPUT" ]; then
        echo "Warning: Source directory $SRC_INPUT does not exist. Skipping."
        continue
    fi

    # Create the target destination directories
    mkdir -p "$DEST_DIR/input"

    # Copy the evaluations (gpt-4o and gpt-5)
    if [ -d "$SRC_EVALS" ]; then
        echo "Copying evaluations..."
        cp -R "$SRC_EVALS" "$DEST_DIR/"
    fi

    # Identify untracked/ignored files (what git clean would remove)
    UNTRACKED=$(git clean -ndx "$SRC_INPUT" 2>/dev/null | sed 's/Would remove //')

    # Identify modified tracked files (e.g., modified scripts in replication_data)
    MODIFIED=$(git diff HEAD --name-only "$SRC_INPUT" 2>/dev/null)

    # Combine lists and copy files over
    { echo "$UNTRACKED"; echo "$MODIFIED"; } | while IFS= read -r filepath; do
        # Make sure the file/folder actually exists and path isn't empty
        if [ -n "$filepath" ] && [ -e "$filepath" ]; then
            
            # Get the path relative to the input folder (e.g., _artifacts/)
            REL_PATH="${filepath#$SRC_INPUT/}"

            # Ensure the parent directory exists in the destination
            mkdir -p "$(dirname "$DEST_DIR/input/$REL_PATH")"

            # Copy recursively (handles both files and whole directories)
            # the use of -R to grab entire directories correctly
            cp -R "$filepath" "$DEST_DIR/input/$REL_PATH"
            
            echo "Copied: input/$REL_PATH"
        fi
    done
    
    echo "Finished ID: $id"
done

echo "----------------------------------------"
echo "All done! Your files have been moved to results/native/..."
