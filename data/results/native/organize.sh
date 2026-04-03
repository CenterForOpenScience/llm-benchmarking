for target_dir in */gpt-4o; do
    echo "Structuring $target_dir..."
    
    # 1. Create the new nested evaluator directory
    mkdir -p "$target_dir/evals/gpt-4o"
    
    # 2. If the generic 'llm_eval' folder exists, move its contents and remove it
    if [ -d "$target_dir/llm_eval" ]; then
        mv "$target_dir/llm_eval/"* "$target_dir/evals/gpt-4o/" 2>/dev/null
        rmdir "$target_dir/llm_eval"
    fi
    
    # 3. Move any loose evaluation files from the root into the new evaluator folder
    # (We use 2>/dev/null to suppress errors if a specific file doesn't exist in a folder)
    mv "$target_dir"/*llm_eval* "$target_dir/evals/gpt-4o/" 2>/dev/null
    mv "$target_dir"/eval_summary.json "$target_dir/evals/gpt-4o/" 2>/dev/null
    mv "$target_dir"/evaluate_execute.log "$target_dir/evals/gpt-4o/" 2>/dev/null
    mv "$target_dir"/interpret_eval.log "$target_dir/evals/gpt-4o/" 2>/dev/null
done

echo "Done! Run 'tree .' to verify."
