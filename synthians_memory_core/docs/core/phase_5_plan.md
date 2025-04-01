The following provides a phased implementation plan, including code snippets, edge case considerations, testing guidance, and the AI IDE Developer Prompt, using the provided LM Studio details.

---

## Synthians Cognitive Architecture: Phased Implementation Plan (Phase 5)

**Overall Goal:** Transition from the stable Phase 4.6 architecture to Phase 5, enabling adaptive intelligence through dynamic variant selection, LLM-guided memory operations (via LM Studio), adaptive attention heuristics, and enhanced diagnostics.

**Models (LM Studio):**

*   **Real-time Guidance:** `hugging-quants/llama-3.2-1b-instruct` (Assumed loaded and available at the LM Studio endpoint).
*   **Async/Dream Tasks:** `Qwen/Qwen1.5-0.5B-Chat` (Corrected model name for clarity, assumed loaded/available).
*   **LM Studio Endpoint:** `http://127.0.0.1:1234/v1/chat/completions` (or configured URL).

---

### **Phase 5.0: Foundation & Refactoring**

*   **Objective:** Prepare the CCE and Variant codebase for adaptive integration.
*   **Key Tasks:**
    1.  Refactor `ContextCascadeEngine.set_variant` for internal use.
    2.  Introduce `attention_hints` parameter stub to variant processing methods.
    3.  Add a CCE endpoint for accessing recent response metrics.
*   **Code Snippets:**

    *   **CCE: Internal Variant Switching:**
        ```python
        # orchestrator/context_cascade_engine.py
        class ContextCascadeEngine:
            # ... (existing init) ...

            async def _switch_variant_internal(self, new_variant_type: TitansVariantType, reset_nm: bool = False):
                """Internal method to switch variant without dev mode check."""
                logger.info(f"Internal variant switch to: {new_variant_type.value}")
                # 1. Acquire Lock (ensure no processing is ongoing) - Use a dedicated internal lock if needed?
                async with self.processing_lock: # Reuse existing lock for simplicity initially
                    # 2. Flush Context
                    context_size_before = len(self.sequence_context_manager)
                    self.sequence_context_manager.clear()
                    self.sequence_context.clear() # Clear legacy if still used
                    logger.info(f"Internal switch: Flushed context ({context_size_before} entries).")
                    # 3. Update Active Variant Type
                    previous_variant = self.active_variant_type.value
                    self.active_variant_type = new_variant_type
                    # 4. Reconfigure Processor
                    self.variant_processor = None # Clear old processor
                    try:
                        await self._configure_attention_and_variant() # Re-init based on new type
                        reconfigured = self.variant_processor is not None or new_variant_type == TitansVariantType.NONE
                        logger.info(f"Internal switch: Reconfigured processor for {new_variant_type.value}. Success: {reconfigured}")
                    except Exception as e:
                        logger.error(f"Internal switch: Error reconfiguring for {new_variant_type.value}: {e}")
                        # Potentially revert active_variant_type or handle error state
                    # 5. Reset Neural Memory (Optional)
                    if reset_nm:
                        # Call NM /init endpoint
                        await self._make_request(self.neural_memory_url, "/init", method="POST", payload={"force_reset": True})
                        logger.info("Internal switch: Requested Neural Memory reset.")
                # 6. Release Lock

            # Internal method might not need to return full API response dict
            return {"success": True, "switched_to": new_variant_type.value, "previous": previous_variant}

            async def set_variant(self, variant_type_str: str, reset_neural_memory: bool = False) -> Dict[str, Any]:
                """Set the active Titans variant at runtime (DevMode)."""
                dev_mode_enabled = os.environ.get("CCE_DEV_MODE", "false").lower() in ("true", "1", "yes")
                # ... (rest of existing dev mode checks and validation) ...
                if not dev_mode_enabled:
                     raise RuntimeError("Cannot switch variants: CCE_DEV_MODE is not enabled")
                if self.processing_lock.locked():
                     raise RuntimeError("Cannot switch variants while processing")
                # ... (validate variant_type_str) ...
                new_variant_type = TitansVariantType(variant_type_str.upper())
                if new_variant_type == self.active_variant_type:
                     # ... (return unchanged status) ...

                # Call the internal method
                result = await self._switch_variant_internal(new_variant_type, reset_neural_memory)

                # Log audit trail externally
                # ... (log to file as before) ...

                # Return API response
                return {**result, "dev_mode": dev_mode_enabled, "status": "switched" if result["success"] else "error"}

        ```

    *   **Variants: Add `attention_hints` stub:**
        ```python
        # orchestrator/titans_variants.py
        class TitansVariantBase:
            # ...
            async def process_input(self, memory_id: str, x_t: Any, k_t: Any,
                                v_t: Any, q_t: Any, y_t: Any,
                                attention_hints: Optional[Dict[str, Any]] = None # <-- ADDED
                               ) -> Dict[str, Any]:
                 """Process input through the variant's logic."""
                 if attention_hints:
                     logger.debug(f"{self.name}: Received attention hints: {attention_hints}")
                 # ... (existing context storage) ...
                 # Base implementation just returns y_t unchanged
                 return {"y_t_final": y_t, "metrics": {}, "success": True}

        # Update MAC, MAG, MAL process_input/calculate_v_prime methods similarly
        # Example in MACVariant:
        class MACVariant(TitansVariantBase):
             async def process_input(self, ..., attention_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                 # ...
                 try:
                     # Use hints if provided to adjust attention params/masking
                     focus = attention_hints.get('focus', 'default') if attention_hints else 'default'
                     # ... (modify attention calculation based on focus) ...
                 # ...
                 except Exception as e:
                     # ... handle errors ...
                     metrics["error"] = f"Error processing hints: {str(e)}"
                     # Return existing y_t as fallback
                     return {"y_t_final": y_t, "metrics": metrics, "success": False} # Indicate hint processing error?
        ```

    *   **CCE: Metrics Endpoint:**
        ```python
        # orchestrator/server.py
        from collections import deque
        from fastapi import Query

        # Add a deque to CCE class to store recent responses
        class ContextCascadeEngine:
            def __init__(self, ...):
                # ...
                self.recent_responses_buffer: deque = deque(maxlen=50) # Store last 50 responses

            async def process_new_input(self, ...):
                # ... (at the end, before returning response)
                self.recent_responses_buffer.append(response) # Store the full response
                return response

        # Add endpoint to server.py
        @app.get("/metrics/recent_cce_responses")
        async def get_recent_responses(limit: int = Query(10, ge=1, le=50)):
            """Retrieve the last N CCE response objects."""
            orchestrator = get_orchestrator()
            # Return responses from the buffer
            responses = list(orchestrator.recent_responses_buffer)
            return responses[-limit:]
        ```

*   **Edge Cases:**
    *   Ensure internal variant switching handles locks correctly to prevent race conditions.
    *   Test behavior if `_configure_attention_and_variant` fails during an internal switch.
    *   Confirm variants handle `attention_hints=None` gracefully.
    *   Verify `/metrics/recent_cce_responses` handles empty buffer.
*   **Testing:**
    *   Unit test `_switch_variant_internal` logic (mocking reconfiguration).
    *   Integration test internal switching via a dedicated test endpoint or by modifying `process_new_input` temporarily.
    *   Test `/metrics/recent_cce_responses` endpoint returns correct data.
*   **Key Files:** `orchestrator/context_cascade_engine.py`, `orchestrator/titans_variants.py`, `orchestrator/server.py`.

---

### **Phase 5.1: Diagnostics Dashboard**

*   **Objective:** Implement the `variant_diagnostics_dashboard.py` CLI tool to monitor CCE variant metrics in real-time.
*   **Key Tasks:**
    1.  Implement data fetching using the new `/metrics/recent_cce_responses` endpoint.
    2.  Implement parsing logic for the standardized `variant_output`.
    3.  Implement display logic using `rich.Table` to show active variant and its specific metrics.
    4.  Add basic error display.
    5.  Implement the main polling loop and CLI arguments.
*   **Code Snippet (Dashboard Fetching & Parsing):**
    ```python
    # tools/variant_diagnostics_dashboard.py
    import requests, json, time, argparse
    from rich.console import Console
    from rich.table import Table
    from collections import deque # Needed if calculating averages

    console = Console()
    # Store history for trend analysis (optional)
    metrics_history = deque(maxlen=100)

    def fetch_data_from_cce(cce_url, limit=1):
        try:
            response = requests.get(f"{cce_url}/metrics/recent_cce_responses", params={"limit": limit}, timeout=5)
            response.raise_for_status()
            data = response.json()
            # Get the latest response if multiple are returned
            return data[-1] if data else None
        except Exception as e:
            console.print(f"[red]Error fetching CCE data: {e}[/red]")
            return None

    def parse_variant_output(data):
        # ... (As defined previously) ...
        if not data or "variant_output" not in data: return "UNKNOWN", {}
        vo = data["variant_output"]
        vt = vo.get("variant_type", "UNKNOWN")
        vk = vt.lower()
        metrics = vo.get(vk, {})
        return vt, metrics

    def display_metrics(data, variant_type, metrics):
        table = Table(title=f"Variant Diagnostics ({variant_type}) @ {data.get('timestamp', 'N/A')}")
        # ... (Add columns and rows as before) ...
        console.print(table)
        # Optionally display trends from metrics_history

    def main(cce_url, interval):
        while True:
            console.clear()
            console.print(f"[bold cyan]Variant Diagnostics Dashboard ({cce_url}) - Refreshing every {interval}s[/bold cyan]")
            data = fetch_data_from_cce(cce_url, limit=1)
            if data:
                variant_type, metrics = parse_variant_output(data)
                metrics_history.append({"ts": data.get("timestamp"), "type": variant_type, **metrics})
                display_metrics(data, variant_type, metrics)
            else:
                console.print("[yellow]No data received from CCE.[/yellow]")
            time.sleep(interval)
    # ... (argparse and main execution block) ...
    ```
*   **Edge Cases:** CCE endpoint down, invalid JSON response, missing `variant_output` key, empty metrics dictionary.
*   **Testing:** Run the dashboard tool against a running CCE; verify it displays correct info for different active variants; test connection error handling.
*   **Key Files:** `tools/variant_diagnostics_dashboard.py`, `orchestrator/server.py` (for endpoint).

---

### **Phase 5.2: Variant Selector Module**

*   **Objective:** Implement the rule-based `VariantSelector` to replace static variant configuration.
*   **Key Tasks:**
    1.  Create `orchestrator/variant_selector.py` with `VariantSelector` class.
    2.  Implement `select_variant` method with initial rules based on query keywords, metadata hints (`task_type`, `emotion`), and potentially recent NM performance (pass avg loss/grad).
    3.  Integrate the `VariantSelector` into `CCE.process_new_input`.
    4.  Trigger internal variant switching using the refactored mechanism from Phase 5.0.
*   **Code Snippets:**

    *   **Variant Selector Logic:**
        ```python
        # orchestrator/variant_selector.py
        from .titans_variants import TitansVariantType
        from typing import Dict, Any, Optional

        class VariantSelector:
            def __init__(self, high_surprise_threshold=0.5, low_surprise_threshold=0.1):
                self.high_surprise_threshold = high_surprise_threshold
                self.low_surprise_threshold = low_surprise_threshold

            def select_variant(self, query: Optional[str], metadata: Dict[str, Any],
                               nm_performance: Dict[str, float], llm_hint: Optional[str]) -> TitansVariantType:
                """Selects the best variant based on context and performance."""

                # 1. Check LLM Hint (Highest Priority)
                if llm_hint and llm_hint in TitansVariantType.__members__:
                    return TitansVariantType(llm_hint)

                # 2. Check Metadata Hints
                if metadata.get("task_type") == "summarize": return TitansVariantType.MAC
                if metadata.get("task_type") == "causal_reasoning": return TitansVariantType.MAL
                if metadata.get("priority") == "background": return TitansVariantType.NONE

                # 3. Check Performance Metrics
                avg_loss = nm_performance.get("avg_loss", 0.0)
                if avg_loss > self.high_surprise_threshold:
                    return TitansVariantType.MAG # High surprise -> Adapt learning parameters

                # 4. Check Query Keywords (Example)
                query_lower = query.lower() if query else ""
                if "explain why" in query_lower or "cause of" in query_lower:
                    return TitansVariantType.MAL # Causal reasoning
                if "remember when" in query_lower or "recall events" in query_lower:
                    return TitansVariantType.MAC # Sequential recall

                # 5. Default Logic
                if avg_loss < self.low_surprise_threshold:
                    return TitansVariantType.NONE # Low surprise, be efficient
                else:
                    return TitansVariantType.MAC # Default to MAC for general context

        ```

    *   **CCE Integration:**
        ```python
        # orchestrator/context_cascade_engine.py
        from .variant_selector import VariantSelector # Import

        class ContextCascadeEngine:
            def __init__(self, ...):
                # ...
                self.variant_selector = VariantSelector()
                self.nm_performance_history = deque(maxlen=20) # Track recent loss/grad

            async def process_new_input(self, ...):
                # ... (Steps 1-3: Store, Projections, LLM Router) ...
                advice = router_response.get('advice', {}) # Get LLM advice

                # Calculate recent performance (simple average)
                avg_loss = np.mean([p.get('loss', 0.0) for p in self.nm_performance_history if p.get('loss') is not None]) if self.nm_performance_history else 0.0
                nm_perf = {"avg_loss": avg_loss}

                # ---> Step 4 & 5: Select and Switch Variant <---
                selected_variant_type = self.variant_selector.select_variant(
                    query=step_context["content"],
                    metadata=step_context["metadata"],
                    nm_performance=nm_perf,
                    llm_hint=advice.get('variant_hint')
                )
                if selected_variant_type != self.active_variant_type:
                    await self._switch_variant_internal(selected_variant_type, reset_nm=False) # Don't reset NM by default

                # ---> Step 6: Variant Pre-Update <---
                # Pass attention hints from LLM advice
                attention_hints = {"focus": advice.get("attention_focus")} if advice.get("attention_focus") else None
                if self.variant_processor and self.active_variant_type in [TitansVariantType.MAG, TitansVariantType.MAL]:
                    variant_pre_result = await self._apply_variant_pre_update(step_context, attention_hints=attention_hints) # Pass hints
                    # ...

                # ---> Step 7: Update NM <---
                update_resp = await self._update_neural_memory(step_context)
                # Add loss/grad to performance history
                if update_resp.get("success"):
                    self.nm_performance_history.append({
                        "loss": update_resp.get("loss"),
                        "grad_norm": update_resp.get("grad_norm")
                    })

                # ---> Step 8: Apply Boost <---
                boost_modifier = advice.get('boost_score_mod', 0.0) # Get LLM boost adjustment
                feedback_resp = await self._apply_quickrecal_boost(step_context, quickrecal_initial, boost_modifier=boost_modifier) # Pass modifier

                # ---> Step 10: Variant Post-Update (MAC) <---
                if self.variant_processor and self.active_variant_type == TitansVariantType.MAC:
                     variant_post_result = await self._apply_variant_post_retrieval(step_context, attention_hints=attention_hints) # Pass hints
                     # ...

                # ... (Steps 11, 12: History, Response) ...
        ```
*   **Edge Cases:** No history for performance metrics, LLM hint invalid, `select_variant` fails, internal switch fails.
*   **Testing:** Unit test `VariantSelector` rules. Integration test CCE with different inputs/metadata designed to trigger specific variants; verify the correct variant is activated (check logs/`variant_output`) and that `_switch_variant_internal` is called.
*   **Key Files:** `orchestrator/variant_selector.py` (new), `orchestrator/context_cascade_engine.py`.

---

### **Phase 5.3: LLM Memory Guidance**

*   **Objective:** Integrate `MemoryLLMRouter` using LM Studio to guide memory operations.
*   **Key Tasks:**
    1.  Create `orchestrator/memory_logic_proxy.py` with `MemoryLLMRouter`.
    2.  Implement `request_llama_guidance` method:
        *   Format prompt using the designed template.
        *   Make async HTTP POST call to LM Studio (`/v1/chat/completions`) using `aiohttp`.
        *   Include `response_format` for structured JSON output.
        *   Parse the JSON advice from the response content.
        *   Handle errors (LM Studio down, invalid response, timeout).
    3.  Integrate the router call into `CCE.process_new_input` (as shown in Phase 5.2).
    4.  Modify CCE logic (boost calculation, potentially storage decision, hint forwarding) based on the LLM's advice.
*   **Code Snippets:**

    *   **MemoryLLMRouter:**
        ```python
        # orchestrator/memory_logic_proxy.py
        import aiohttp
        import json
        import logging
        from typing import Dict, Any, Optional

        logger = logging.getLogger(__name__)

        class MemoryLLMRouter:
            DEFAULT_PROMPT_TEMPLATE = """SYSTEM:
You are a memory decision-making assistant... [Your Full Prompt Template Here] ...Now return your JSON decision block:"""

            DEFAULT_LLM_SCHEMA = {
                  "name": "memory_decision",
                  "strict": "true", # Enforce schema strictly
                  "schema": {
                      "type": "object",
                      "properties": {
                          "store": {"type": "boolean"},
                          "metadata_tags": {"type": "array", "items": {"type": "string"}},
                          "boost_score_mod": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                          "variant_hint": {"type": "string", "enum": ["NONE", "MAC", "MAG", "MAL"]},
                          "attention_focus": {"type": "string", "enum": ["recency", "relevance", "emotional", "broad"]},
                          "notes": {"type": "string"}
                      },
                      "required": ["store", "metadata_tags", "boost_score_mod", "variant_hint", "attention_focus", "notes"]
                  }
              }

            def __init__(self, mode="llmstudio", llama_endpoint="http://127.0.0.1:1234/v1/chat/completions", llama_model="hugging-quants/llama-3.2-1b-instruct"):
                self.mode = mode
                self.llama_endpoint = llama_endpoint
                self.llama_model = llama_model
                self.session = None
                logger.info(f"MemoryLLMRouter initialized in '{mode}' mode for model '{llama_model}' at '{llama_endpoint}'")

            async def _get_session(self):
                if self.session is None or self.session.closed:
                    self.session = aiohttp.ClientSession()
                return self.session

            async def close_session(self):
                if self.session:
                    await self.session.close()
                    self.session = None

            async def request_llama_guidance(self, user_input: str, nm_feedback: Dict, metadata: Dict) -> Dict[str, Any]:
                """Requests guidance from the LLAMA model via LM Studio."""
                if self.mode != "llmstudio":
                    logger.warning("LLM Router not in llmstudio mode, returning default advice.")
                    return self._default_advice()

                prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
                    user_input=user_input or "[No Input Text]",
                    loss=nm_feedback.get('loss', 'N/A'),
                    grad_norm=nm_feedback.get('grad_norm', 'N/A'),
                    retrieved_memories_summary="[Summary Placeholder]", # TODO: Summarize retrieval results
                    variant_type=metadata.get('variant_type', 'N/A'),
                    emotion=metadata.get('emotion', 'neutral'),
                    task_type=metadata.get('task_type', 'unknown'),
                    context_signal=metadata.get('context_signal', 'none'),
                    entry_1="[History Placeholder 1]", # TODO: Populate recent history
                    entry_2="[History Placeholder 2]",
                    entry_3="[History Placeholder 3]"
                )

                payload = {
                    "model": self.llama_model,
                    "messages": [
                        # System prompt is embedded in the user prompt template
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3, # Low temp for deterministic advice
                    "max_tokens": 256, # Limit response size
                    "stream": False,
                    "response_format": { # Request structured JSON output
                        "type": "json_schema",
                        "json_schema": self.DEFAULT_LLM_SCHEMA
                    }
                }

                session = await self._get_session()
                try:
                    async with session.post(self.llama_endpoint, json=payload, timeout=15) as response:
                        if response.status == 200:
                            resp_json = await response.json()
                            content_str = resp_json["choices"][0]["message"]["content"]
                            try:
                                advice = json.loads(content_str)
                                # Validate advice against schema (basic check)
                                if all(k in advice for k in self.DEFAULT_LLM_SCHEMA["schema"]["required"]):
                                    logger.info(f"LLM Guidance Received: {advice}")
                                    return advice
                                else:
                                    logger.error(f"LLM response missing required keys: {content_str}")
                                    return self._default_advice("LLM response missing keys")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse LLM JSON response: {content_str}")
                                return self._default_advice("LLM JSON parse error")
                        else:
                            error_text = await response.text()
                            logger.error(f"LM Studio API error ({response.status}): {error_text}")
                            return self._default_advice(f"LM Studio API error {response.status}")
                except asyncio.TimeoutError:
                    logger.error("Timeout connecting to LM Studio.")
                    return self._default_advice("LM Studio timeout")
                except aiohttp.ClientConnectorError as e:
                     logger.error(f"Connection error to LM Studio: {e}")
                     return self._default_advice("LM Studio connection error")
                except Exception as e:
                    logger.error(f"Error requesting LLM guidance: {e}", exc_info=True)
                    return self._default_advice(str(e))

            def _default_advice(self, error_msg="Default advice triggered"):
                """Returns default guidance when LLM call fails."""
                return {
                    "store": True, "metadata_tags": ["error_llm_guidance"],
                    "boost_score_mod": 0.0, "variant_hint": "NONE",
                    "attention_focus": "broad", "notes": error_msg
                }

            # Add request_qwen_dream_task later for Phase 5.5
        ```
    *   **CCE: Apply Advice:**
        ```python
        # orchestrator/context_cascade_engine.py
        async def _apply_quickrecal_boost(self, ..., boost_modifier=0.0):
             # ... calculate base boost from loss/grad_norm ...
             final_boost = base_boost + boost_modifier # Apply LLM modifier
             final_boost = max(0.0, final_boost) # Ensure non-negative
             # ... make API call with final_boost ...
        ```

*   **Edge Cases:** LM Studio unavailable/slow, LLM returns malformed JSON, LLM advice conflicts with other logic, network errors.
*   **Testing:** Unit test `MemoryLLMRouter` (mock `aiohttp.post`). Integration test CCE with a mock LM Studio server returning predefined advice; verify CCE applies the advice correctly (e.g., variant hint used, boost modified). Test error handling when LM Studio is down.
*   **Key Files:** `orchestrator/memory_logic_proxy.py` (new), `orchestrator/context_cascade_engine.py`, LM Studio configuration/setup.

---

### **Phase 5.4: Adaptive Attention Heuristics**

*   **Objective:** Implement simple context-based adjustments to attention mechanisms.
*   **Key Tasks:**
    1.  Modify CCE to determine appropriate `max_length` for `SequenceContextManager` based on task type or LLM hint.
    2.  Modify CCE to construct `attention_hints` dictionary based on task/LLM advice.
    3.  Update `MACVariant`, `MAGVariant`, `MALVariant` `process_input` (or internal attention methods) to *use* the `attention_hints` (e.g., adjust masking, temperature, or simply log the hint for now).
*   **Code Snippets:**

    *   **CCE: Context Length Adjustment:**
        ```python
        # orchestrator/context_cascade_engine.py
        class ContextCascadeEngine:
            async def process_new_input(self, ...):
                # ... (After getting LLM advice or inferring task type) ...
                task_type = advice.get('task_type', 'general')
                if task_type == 'summarize':
                    self.sequence_context_manager.max_length = 100 # Increase for summary
                else:
                    self.sequence_context_manager.max_length = self.sequence_context_length # Default
                # ...
                attention_hints = {"focus": advice.get("attention_focus", "broad")}
                # ... (Pass hints to variant processing) ...
        ```
*   **Edge Cases:** Rapid task switching causing frequent `max_length` changes, hints not recognized by variants.
*   **Testing:** Integration test CCE with inputs designed to trigger different task types; verify `SequenceContextManager.max_length` changes accordingly (via logging or a status endpoint). Verify `attention_hints` are passed and potentially logged by variants.
*   **Key Files:** `orchestrator/context_cascade_engine.py`, `orchestrator/titans_variants.py`.

---

### **Phase 5.5: Async "Dream" Tasks (Placeholder)**

*   **Objective:** Integrate Qwen model for offline/async memory analysis.
*   **Key Tasks:**
    1.  Implement `MemoryLLMRouter.request_qwen_dream_task`.
    2.  Create a mechanism (e.g., scheduler, separate process) to trigger these tasks during idle periods.
    3.  Define specific dream tasks (summarization, contradiction finding, abstraction).
    4.  Determine how results feed back into the Memory Core (e.g., storing summaries as new `MemoryEntry`s).
*   **Status:** Deferred. Focus on real-time adaptive loop first.

---

## AI IDE Developer Prompt

```text
**Role:** You are an expert AI developer specializing in cognitive architectures, specifically the Synthians project. You have deep knowledge of the Memory Core, Neural Memory (Titans-based), Context Cascade Engine (CCE), and Titans Variants (MAC, MAG, MAL).

**Context:** The project has just completed Phase 4.6. The core bi-hemispheric loop is stable, tested, and features standardized variant metrics output from the CCE (`variant_output` field). We are now transitioning to Phase 5, focusing on adding adaptive intelligence.

**Phase 5 Goals:**
1. Implement dynamic, context-aware **Variant Selection** within the CCE, replacing static configuration.
2. Integrate a **Memory Logic LLM** (LLAMA 3.21B via LM Studio) to guide memory storage, tagging, and scoring based on real-time context.
3. Build a **Diagnostics Dashboard** (CLI initially) to monitor the CCE's variant performance using the standardized metrics.
4. Introduce simple **Adaptive Attention Heuristics** (e.g., context length modulation).
5. (Future) Integrate **Async "Dream" Tasks** using Qwen2.5B via LM Studio.

**Current Task:** Assist in implementing the **Phased Plan for Phase 5**. We will proceed phase by phase (5.0, 5.1, etc.).

**Instructions:**
1.  **Code Generation:** Generate Python code snippets for the specified tasks within each phase (e.g., `_switch_variant_internal`, `VariantSelector` rules, `MemoryLLMRouter` LM Studio calls, dashboard display logic). Adhere to the project's coding style (async, type hints, logging).
2.  **Refactoring:** Provide specific recommendations and code examples for refactoring existing components (like `set_variant`, variant `process_input`) to support Phase 5 features.
3.  **Integration:** Detail how new modules (`VariantSelector`, `MemoryLLMRouter`) should be integrated into the `ContextCascadeEngine`'s `process_new_input` flow.
4.  **Edge Cases & Robustness:** Identify potential edge cases, error conditions (e.g., LM Studio unavailable, invalid LLM response), and suggest robust handling mechanisms (timeouts, fallbacks, validation).
5.  **Testing:** Suggest specific unit and integration test cases needed to validate each phase's implementation.
6.  **LM Studio Details:** Utilize the provided LM Studio endpoint (`http://127.0.0.1:1234/v1/chat/completions`), models (`hugging-quants/llama-3.2-1b-instruct`, `Qwen/Qwen1.5-0.5B-Chat`), and parameters (especially `response_format` with `json_schema` for structured output from LLAMA).
7.  **Clarity & Modularity:** Ensure generated code and integration points are clear, modular, and maintainable.

**Start with Phase 5.0: Foundation & Refactoring.** Provide the refactored code for `set_variant` / `_switch_variant_internal` in CCE, show how to add the `attention_hints` stub to `TitansVariantBase.process_input`, and outline the new CCE metrics endpoint (`/metrics/recent_cce_responses`).
```