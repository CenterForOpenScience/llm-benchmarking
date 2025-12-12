import os
import re
import sys
import time
import json
import logging
import tiktoken
from openai import OpenAI
from core.utils import get_logger
from core.constants import API_KEY, GENERATE_REACT_CONSTANTS

logger, formatter = get_logger()
client = OpenAI(api_key=API_KEY) 

ACTION_PATTERNS = [
    # Single-line Action
    re.compile(r'(?mi)^\s*Action:\s*([a-z0-9_]+)\s*:\s*(.+)$'),
    # Code-fenced arguments (``` or ```json)
    re.compile(r'(?mis)^\s*Action:\s*([a-z0-9_]+)\s*:\s*```(?:json)?\s*(.*?)\s*```'),
    # Bolded "Action:" some models produce
    re.compile(r'(?mis)\*\*Action:\*\*\s*([a-z0-9_]+)\s*:\s*(.+?)\s*(?:\n|$)')
]

def _extract_action(text: str):
    for pat in ACTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1), m.group(2).strip()
    return None

def update_metadata(study_path: str, stage: str, data: dict):
    """
    Updates metadata.json in the study_path with metrics for a specific stage.
    """
    meta_path = os.path.join(study_path, "metadata.json")
    
    # Load existing or create new
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}

    # Update the specific stage
    metadata[stage] = data

    # Write back
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated metadata for {stage} in {meta_path}")

class Agent:
    def __init__(self, system="", session_state=None):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
        self.session_state = session_state or {}

        self._tpm_window_start = time.time()
        self._tpm_tokens = 0  # tokens used since last reset

    def __call__(self, message):
        content, usage = self.execute(message)
        return content, usage

    def execute(self, message=None):
        if message:
            self.messages.append({"role": "user", "content": message})
        # simple 30k TPM limiter
        now = time.time()
        # reset window every 60s
        if now - self._tpm_window_start >= 60:
            self._tpm_window_start = now
            self._tpm_tokens = 0

        # predict tokens for this call: prompt + a small completion reserve
        prompt_tokens = self.count_current_tokens()
        completion_reserve = 1024  # keep it small and simple
        predicted_total = prompt_tokens + completion_reserve

        if self._tpm_tokens + predicted_total > 30000:
            # gentle delay (20–30s); then reset counters
            print("going to sleep...zZZ")
            time.sleep(25)
            self._tpm_window_start = time.time()
            self._tpm_tokens = 0

        # actual call
        completion = client.chat.completions.create(
            model=self.model if hasattr(self, "model") else "gpt-4o",
            temperature=0,
            messages=self.messages,
            max_tokens=completion_reserve,  # optional; keeps outputs bounded
        )

        usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            usage = completion.usage
            if usage:
                usage_stats["prompt_tokens"] = usage.prompt_tokens
                usage_stats["completion_tokens"] = usage.completion_tokens
                usage_stats["total_tokens"] = usage.total_tokens
        except Exception:
            # Fallback to predicted if API doesn't return usage
            usage_stats["total_tokens"] = predicted_total

        self._tpm_tokens += usage_stats["total_tokens"]
        
        result_content = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": result_content})

        return result_content, usage_stats

    def _execute_tool_call(self, known_actions, action, action_input_str):
        """
        Executes a tool call by parsing the input string as JSON.
        """
        tool_func = known_actions[action]
        
        try:
            # The ONLY parsing step you need. It correctly handles quotes,
            # escapes, and complex objects. No more codecs.
            # parsed_args = json.loads(repair_json(action_input_str))
            parsed_args =  json.loads(action_input_str.strip())
            
            print(f"DEBUG: Parsed args for '{action}': {parsed_args}, Type: {type(parsed_args)}")

            # Your existing logic for calling the function is good, let's keep it.
            if isinstance(parsed_args, dict):
                # For tools expecting keyword arguments, e.g., func(**{"path": "...", "content": "..."})
                if "dataset" in action:
                    observation = tool_func(self.session_state, **parsed_args)
                else:
                    observation = tool_func(**parsed_args)
            else:
                # For tools expecting a single positional argument, e.g., func("my_file.txt")
                if "dataset" in action:
                    observation = tool_func(self.session_state, parsed_args)
                else:
                    observation = tool_func(parsed_args)
            
            return observation

        except json.JSONDecodeError:
            return f"Error: The tool input was not valid JSON. Please check your formatting. Input received: {action_input_str}"
        except Exception as e:
            return f"Error while executing tool '{action}': {e}"


    def _get_encoding(self):
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def count_current_tokens(self) -> int:
        """
        Rough count of tokens of self.messages (input side only).
        Uses tiktoken if available; otherwise ~4 chars/token heuristic.
        """
        enc = self._get_encoding()

        def cnt(s: str) -> int:
            if enc:
                return len(enc.encode(s))
            return max(1, len(s) // 4)

        total = 0
        for m in self.messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # Only count text parts for multimodal messages
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            elif not isinstance(content, str):
                content = str(content)
            total += cnt(role + ": " + content) + 6  # small overhead
        total += 2
        return total

def run_react_loop(system_prompt: str, known_actions: dict, question: str, *,
                   max_turns: int = 30, session_state=None, on_final=None, log_turns: bool=True,
                   study_path: str = None, stage_name: str = None, checkpoint_map: dict = None):
    
    bot = Agent(system_prompt, session_state=session_state or {})    
    next_prompt = question
    
    start_time = time.time()

    total_tokens_used = 0
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0

    turn_metrics = []
    
    # Checkpoint logic
    current_checkpoint = "0. Initialization"
    checkpoint_stats = {} 

    for i in range(max_turns):
        turn_start = time.time()
        
        if log_turns:
            disp = next_prompt if len(next_prompt) <= 2000 else next_prompt[:2000] + "\n... (truncated)"
            logger.info(f"\n--- Turn {i+1} ---")
            logger.info(f"***Agent input: {disp}")

            result, usage = bot(next_prompt)

            turn_prompt = usage.get("prompt_tokens", 0)
            turn_completion = usage.get("completion_tokens", 0)
            turn_total = usage.get("total_tokens", 0)

            total_prompt_tokens_used += turn_prompt
            total_completion_tokens_used += turn_completion
            total_tokens_used += turn_total

        if log_turns:
            logger.info(f"***Agent output:\n{result}")

        action = None
        is_final = False
        
        # Check for Final Answer
        if "Answer:" in result:
            is_final = True
            current_checkpoint = "8. Final Output & Parsing"
        else:
            matched = _extract_action(result)            
            if matched:
                action, action_input_str = matched
                
                # update checkpoint if action is specific
                if checkpoint_map and action in checkpoint_map:
                    current_checkpoint = checkpoint_map[action]
                elif not checkpoint_map:
                    current_checkpoint = "Running Action"

        # Handle Success/final answer
        if is_final:
            turn_duration = time.time() - turn_start
            
            # Add final turn stats
            stats = checkpoint_stats.get(current_checkpoint, {"time": 0.0, "tokens": 0, "turns": 0})
            stats["time"] += turn_duration
            stats["tokens"] += turn_tokens
            stats["turns"] += 1
            checkpoint_stats[current_checkpoint] = stats
            
            try:
                answer_match = re.search(r'Answer:\s*(\{.*?\})\s*$', result, re.DOTALL)
                if answer_match:
                    json_answer_str = answer_match.group(1).strip()
                else:
                    json_answer_str = result.split("Answer:", 1)[1].strip()
                    # (Insert your robust JSON cleaning logic here if needed)

                json_start = json_answer_str.find('{')
                json_end = json_answer_str.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_answer_str = json_answer_str[json_start : json_end + 1]

                final_answer = json.loads(json_answer_str)
                logger.info("\n--- Final Answer Found ---")
                
                if on_final: 
                    on_final(final_answer)

                #  save metrics on success
                if study_path and stage_name:
                    total_time = time.time() - start_time
                    metric_data = {
                        "status": "Success",
                        "total_time_seconds": round(total_time, 2),
                        "total_tokens": total_tokens_used,
                        "prompt_tokens": total_prompt_tokens_used,
                        "completion_tokens": total_completion_tokens_used,
                        "total_turns": i + 1,
                        "checkpoint_stats": checkpoint_stats,
                        "turn_history": turn_metrics
                    }
                    update_metadata(study_path, stage_name, metric_data)

                return final_answer

            except Exception as e:
                logger.error(f"Error parsing final answer: {e}")
                return {"error": str(e)}

        # Handle Action
        elif action:
            logger.info(f" -- Running Action: {action} [Checkpoint: {current_checkpoint}]")

            if action not in known_actions:
                update_metadata(study_path, stage_name, {
                    "error": f"Unknown action: {action}",
                    "partial_turns": turn_metrics
                })
                raise Exception(f"Unknown action: {action}")

            observation = bot._execute_tool_call(known_actions, action, action_input_str)
            next_prompt = f"Observation: {observation}"
            
        else:
            if i == 0:
                next_prompt = "Reminder: Follow the Thought -> Action format."
                continue
            return {"error": "Agent did not provide a recognized action."}

        # Record Turn Metrics
        turn_duration = time.time() - turn_start
        
        # Aggregate Checkpoint Stats
        if current_checkpoint not in checkpoint_stats:
            checkpoint_stats[current_checkpoint] = {"time": 0.0,"tokens": 0,"prompt_tokens": 0,"completion_tokens": 0,"turns": 0,}        
        checkpoint_stats[current_checkpoint]["time"] += turn_duration
        checkpoint_stats[current_checkpoint]["tokens"] += turn_total
        checkpoint_stats[current_checkpoint]["prompt_tokens"] += turn_prompt
        checkpoint_stats[current_checkpoint]["completion_tokens"] += turn_completion
        checkpoint_stats[current_checkpoint]["turns"] += 1

        turn_metrics.append({
            "turn": i + 1,
            "action": action if action else "None",
            "checkpoint": current_checkpoint,
            "duration_seconds": round(turn_duration, 2),
            "prompt_tokens": turn_prompt,
            "completion_tokens": turn_completion,
            "total_tokens": turn_total,
        })

    # If loop finishes without answer
    logger.warning("Max turns reached.")
    if study_path and stage_name:
        update_metadata(study_path, stage_name, {
            "status": "Failed - Max Turns Reached",
            "total_time_seconds": round(time.time() - start_time, 2),
            "total_tokens": total_tokens_used,
            "prompt_tokens": total_prompt_tokens_used,
            "completion_tokens": total_completion_tokens_used,
            "total_turns": max_turns,
            "checkpoint_stats": checkpoint_stats,
            "turn_history": turn_metrics
        })
        
    return {"error": "Max turns reached without a final answer."}

def save_output(extracted_json, study_path, filename: str = "replication_info.json", stage_name: str = "design"):
    os.makedirs(study_path, exist_ok=True)
    out_path = os.path.join(study_path, filename)
    with open(out_path, "w") as f:
        json.dump(extracted_json, f, indent=2)
    logger.info(f"{stage_name.capitalize()} stage output saved to {out_path}")
    return out_path