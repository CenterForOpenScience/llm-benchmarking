IDS=(2 4 7 20 21 22 23 24 26)

for id in "${IDS[@]}"; do
    TARGET="original/$id"
    
    echo "Resetting ID: $id"
    
    # Revert modified files
    git restore "$TARGET" 2>/dev/null
    
    # Nuke untracked files/folders (like _artifacts, _log, evals)
    git clean -fdx "$TARGET" 2>/dev/null
done

echo "All specified IDs reset!"
