for exp_dir in *; do
    # Only process if it's an actual directory (like '1', '10', etc.)
    if [ -d "$exp_dir" ]; then
        echo "Structuring $exp_dir..."
        
        # 1. Create the new nested evaluator directory
        mkdir -p "$exp_dir/evals/gpt-4o"
        
        # 2. If the generic 'llm_eval' folder exists, move its contents and remove it
        if [ -d "$exp_dir/llm_eval" ]; then
            mv "$exp_dir/llm_eval/"* "$exp_dir/evals/gpt-4o/" 2>/dev/null
            rmdir "$exp_dir/llm_eval"
        fi
        
        # 3. Move any loose evaluation files into the new evaluator folder
        mv "$exp_dir"/*llm_eval* "$exp_dir/evals/gpt-4o/" 2>/dev/null
        mv "$exp_dir"/eval_summary.json "$exp_dir/evals/gpt-4o/" 2>/dev/null
        mv "$exp_dir"/evaluate_execute.log "$exp_dir/evals/gpt-4o/" 2>/dev/null
        mv "$exp_dir"/interpret_eval.log "$exp_dir/evals/gpt-4o/" 2>/dev/null
        
        # Clean up the directory if no files were moved to evals/gpt-4o 
        rmdir "$exp_dir/evals/gpt-4o" 2>/dev/null
        rmdir "$exp_dir/evals" 2>/dev/null
    fi
done

echo "Done! Run 'tree 1' to verify."
