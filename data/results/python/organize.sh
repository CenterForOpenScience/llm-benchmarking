# Loop through all experiment folders (1, 2, 3, etc.)
for exp_dir in */; do
    # Loop through all model folders inside the experiment (gpt-4o, gpt-5, o3, etc.)
    for target_dir in "$exp_dir"*; do
        
        # Make sure it is actually a directory
        if [ -d "$target_dir" ]; then
            echo "Structuring $target_dir..."
            
            # 1. Create the new nested evaluator directory
            mkdir -p "$target_dir/evals/gpt-4o"
            
            # 2. If the old 'llm_eval' folder exists, move everything inside it to evals/gpt-4o
            if [ -d "$target_dir/llm_eval" ]; then
                mv "$target_dir/llm_eval/"* "$target_dir/evals/gpt-4o/" 2>/dev/null
                rmdir "$target_dir/llm_eval" 2>/dev/null
            fi
            
            # 3. Move any loose evaluation files (logs, jsons) into evals/gpt-4o
            mv "$target_dir"/*llm_eval* "$target_dir/evals/gpt-4o/" 2>/dev/null
            mv "$target_dir"/eval_summary.json "$target_dir/evals/gpt-4o/" 2>/dev/null
            mv "$target_dir"/evaluate_execute.log "$target_dir/evals/gpt-4o/" 2>/dev/null
            mv "$target_dir"/interpret_eval.log "$target_dir/evals/gpt-4o/" 2>/dev/null
            
            # 4. If a model doesn't have any evaluations yet, remove the empty folders so it stays clean
            rmdir "$target_dir/evals/gpt-4o" 2>/dev/null
            rmdir "$target_dir/evals" 2>/dev/null
        fi
    done
done

echo "Done! The python directory is now fully organized."
