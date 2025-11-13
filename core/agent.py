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
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
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

        # record actual usage if available; else fall back to predicted
        used = predicted_total
        try:
            usage = completion.usage
            if usage:
                pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
                ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
                tt = getattr(usage, "total_tokens", None)
                used = tt if tt is not None else (pt + ct)
        except Exception:
            pass
        self._tpm_tokens += used

        return completion.choices[0].message.content

    def _execute_tool_call(self, known_actions, action, action_input_str):
        """
        Robustly parse tool arguments the LLM may emit as:
          - a dict JSON string:        '{"k":"v"}'
          - a JSON-encoded JSON string: "\"{\\\"k\\\":\\\"v\\\"}\""
          - a raw string (path, etc.)
        Backwards-compatible: if parsing fails, we pass the original string like before.
        """
        s = (action_input_str or "").strip()

        # Strip common code-fence wrapping without being strict
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").strip()

        parsed = None

        # First attempt: parse JSON once
        try:
            tmp = json.loads(s)
            # If the first parse yields a string (JSON-encoded JSON), try a second pass
            if isinstance(tmp, str):
                try:
                    parsed = json.loads(tmp)
                except Exception:
                    parsed = tmp  # leave as string if inner parse fails
            else:
                parsed = tmp
        except json.JSONDecodeError:
            parsed = s  # not JSON → treat as raw string (legacy behavior)

        observation = None
        if isinstance(parsed, dict):
            # Dict -> kwargs (preserve your dataset-session_state special case)
            if "dataset" in action:
                observation = known_actions[action](self.session_state, **parsed)
            else:
                observation = known_actions[action](**parsed)
        else:
            # String -> single positional arg (legacy behavior)
            try:
                if "dataset" in action:
                    observation = known_actions[action](self.session_state, parsed)
                else:
                    observation = known_actions[action](parsed)
            except Exception as e:
                observation = f"ERROR in {action}: {e!s}"

        return observation

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
	max_turns: int = 20, session_state=None, on_final=None, log_turns: bool=True):

    bot = Agent(system_prompt, session_state=session_state or {})    
    next_prompt = question
    for i in range(max_turns):
        if log_turns:
        	disp = next_prompt if len(next_prompt) <= 2000 else next_prompt[:2000] + "\n... (truncated)"
        	logger.info(f"\n--- Turn {i+1} ---")
        	logger.info(f"\n***Agent input: {disp}")
        result = bot(next_prompt)
        if log_turns:
        	logger.info(f"\n***Agent output:\n{result}")
        # Check if the LLM provided a final answer
        if "Answer:" in result:
            try:
                answer_match = re.search(r'Answer:\s*(\{.*?\})\s*$', result, re.DOTALL)
                if answer_match:
                    json_answer_str = answer_match.group(1).strip()
                else:
                    json_answer_str = result.split("Answer:", 1)[1].strip()
                    if json_answer_str.strip().startswith('{') and json_answer_str.strip().endswith('}'):
                            pass # Looks like valid JSON, proceed
                    else:
                        logger.warning(f"Warning: Answer found but doesn't look like clean JSON: {json_answer_str[:200]}...")
                        # Try to find the JSON part more aggressively
                        json_start = json_answer_str.find('{')
                        json_end = json_answer_str.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            json_answer_str = json_answer_str[json_start : json_end + 1]
                        else:
                            raise ValueError("Could not find a valid JSON structure after 'Answer:'")
                json_start = json_answer_str.find('{')
                json_end = json_answer_str.rfind('}')
                if json_start == -1 or json_end == -1 or json_end < json_start:
                    raise ValueError("Could not find a valid JSON object (missing curly braces) after cleaning.")

                final_answer = json.loads(json_answer_str[json_start : json_end + 1])
                logger.info("\n--- Final Answer ---")
                logger.info(json.dumps(final_answer, indent=2))
                if on_final: # save the output on final answer
                	on_final(final_answer)
                logger.info("Process completed")
                return final_answer
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing final JSON answer: {e}")
                logger.error(f"Raw answer: {json_answer_str}")
                return {"error": "Failed to parse final answer JSON"}
            except Exception as e:
                logger.error(f"An error occurred processing final answer: {e}")
                return {"error": str(e)}
        else:
            matched = _extract_action(result)            
            if matched:
                action, action_input_str = matched
                logger.info(f" -- Running Action: {action} with input: {action_input_str}")

                if action not in known_actions:
                    logger.error(f"Unknown action: {action}: {action_input_str}") 
                    raise Exception(f"Unknown action: {action}: {action_input_str}")

                observation = bot._execute_tool_call(known_actions, action, action_input_str)
                next_prompt = f"Observation: {observation}"
            else:
                if i == 0:
                	print("\n\n\nAgent did not produce parseable respoonse\n\n\n")
                	next_prompt = ("Reminder: Follow the Thought → Action → PAUSE → Observation loop. "
                                   "Do not answer directly. Propose your first Action now.")
                	continue
                logger.warning("Agent did not propose an action. Terminating.")

                # If the agent doesn't provide an action or an answer, something is wrong or it's stuck.
                return {"error": "Agent did not provide a recognized action or final answer."}

    print("Max turns reached. Agent terminated without a final answer.")
    return {"error": "Max turns reached without a final answer."}

def save_output(extracted_json, study_path, filename: str = "replication_info.json", stage_name: str = "design"):
    os.makedirs(study_path, exist_ok=True)
    out_path = os.path.join(study_path, filename)
    with open(out_path, "w") as f:
        json.dump(extracted_json, f, indent=2)
    logger.info(f"{stage_name.capitalize()} stage output saved to {out_path}")
    return out_path