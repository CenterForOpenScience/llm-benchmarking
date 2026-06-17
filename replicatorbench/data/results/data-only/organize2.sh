# STEP 1: Flatten any nested nocode folders (like gpt-5_nocode) into the experiment root
for exp_dir in */; do
    for nocode_dir in "$exp_dir"*nocode*; do
        if [ -d "$nocode_dir" ]; then
            echo "Extracting files from $nocode_dir to $exp_dir..."
            
            # Move all contents from the nested folder up to the experiment root
            mv "$nocode_dir"/* "$exp_dir" 2>/dev/null
            
            # Remove the now-empty nested folder
            rmdir "$nocode_dir" 2>/dev/null
        fi
    done
done

# STEP 2: Standardize the evals to sit in evals/gpt-4o/ directly in the experiment root
for exp_dir in */; do
    if [ -d "$exp_dir" ]; then
        echo "Structuring evals for $exp_dir..."
        
        mkdir -p "${exp_dir}evals/gpt-4o"
        
        # 1. Migrate the legacy llm_eval folder if it exists
        if [ -d "${exp_dir}llm_eval" ]; then
            mv "${exp_dir}llm_eval/"* "${exp_dir}evals/gpt-4o/" 2>/dev/null
            rmdir "${exp_dir}llm_eval" 2>/dev/null
        fi
        
        # 2. Move any loose eval files from the experiment root
        mv "${exp_dir}"*llm_eval* "${exp_dir}evals/gpt-4o/" 2>/dev/null
        mv "${exp_dir}"eval_summary.json "${exp_dir}evals/gpt-4o/" 2>/dev/null
        mv "${exp_dir}"evaluate_execute.log "${exp_dir}evals/gpt-4o/" 2>/dev/null
        mv "${exp_dir}"interpret_eval.log "${exp_dir}evals/gpt-4o/" 2>/dev/null
        
        # 3. Clean up the evals folder if it ended up empty
        rmdir "${exp_dir}evals/gpt-4o" 2>/dev/null
        rmdir "${exp_dir}evals" 2>/dev/null
    fi
done

echo "Done! data-only is flattened and standardized."
