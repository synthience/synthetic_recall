#!/usr/bin/env python

import aiohttp
import json
import logging
import asyncio
import time
import os
from typing import Dict, Any, Optional, List
import jsonschema

logger = logging.getLogger(__name__)

class MemoryLLMRouter:
    """
    Interface with LM Studio to get structured advice for memory operations.
    
    This class handles communication with LM Studio API to get AI-guided
    advice for memory processing, variant selection, attention focus, and
    quickrecal score adjustments.
    """
    
    # Define the structured JSON output schema expected from the LLM
    DEFAULT_LLM_SCHEMA = {
        "name": "memory_decision_advice", # Function name for the schema
        "description": "Provides structured advice for memory processing operations.",
        "strict": True, # Enforce schema strictly
        "schema": {
            "type": "object",
            "properties": {
                "store": {
                    "type": "boolean",
                    "description": "Decision whether to store the current memory entry."
                },
                "metadata_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant tags or keywords to add to the memory's metadata."
                },
                "boost_score_mod": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0,
                    "description": "Modifier (-1.0 to 1.0) to apply to the QuickRecal surprise boost. 0 means no change."
                },
                "variant_hint": {
                    "type": "string",
                    "enum": ["NONE", "MAC", "MAG", "MAL"],
                    "description": "Suggested Titans variant for processing the NEXT input."
                },
                "attention_focus": {
                    "type": "string",
                    "enum": ["recency", "relevance", "emotional", "broad", "specific_topic"],
                    "description": "Suggested focus for attention mechanisms (e.g., prioritize recent history, relevance to query, emotional context)."
                },
                "notes": {
                    "type": "string",
                    "description": "Brief reasoning or notes from the assistant."
                },
                "decision_trace": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step-by-step tracing of the decision process used."
                }
            },
            "required": ["store", "metadata_tags", "boost_score_mod", "variant_hint", "attention_focus", "notes"]
        }
    }

    DEFAULT_PROMPT_TEMPLATE = """SYSTEM: 
You are an advanced cognitive process advisor integrated into the Synthians memory system. Your role is to analyze incoming information and provide structured guidance on how it should be processed and stored. Based on the user input, recent memory context, neural memory feedback (surprise), performance metrics, and current system state, return a JSON object conforming EXACTLY to the following schema:

PROMPT VERSION: 5.6.3

```json
{{
  "store": boolean, // Should this memory be stored?
  "metadata_tags": ["tag1", "tag2", ...], // Relevant tags (keywords, topics)
  "boost_score_mod": float, // Adjust surprise boost (-1.0 to 1.0, 0 = no change)
  "variant_hint": "NONE" | "MAC" | "MAG" | "MAL", // Hint for NEXT step's variant
  "attention_focus": "recency" | "relevance" | "emotional" | "broad" | "specific_topic", // Hint for attention mechanism focus
  "notes": "Brief reasoning for decisions.",
  "decision_trace": ["step1", "step2", ...] // Optional tracing of your decision process
}}
```

Prioritize accuracy and consistency. Higher surprise (loss/grad_norm) usually means the input is novel or unexpected, warranting storage and potentially a positive boost modification. 

PERFORMANCE HEURISTICS:
- High surprise (loss/grad_norm > {high_surprise_threshold:.2f}): Consider MAG variant to help adaptation
- Low surprise (loss/grad_norm < {low_surprise_threshold:.2f}): Consider NONE variant for efficiency
- Increasing trend: Prioritize MAG variant to adapt to the changing pattern
- Decreasing trend in moderate range: Consider MAL for refinement
- System confidence level affects how much your advice will be weighted:
  * High confidence: Your advice will be fully applied
  * Moderate confidence: Your advice may be partially scaled down
  * Low confidence: Your advice may be significantly reduced or ignored

USER_INPUT:
{user_input}

METADATA / CONTEXT:
- Task Type: {task_type}
- User Emotion: {emotion}
- Current Variant: {current_variant}

NEURAL MEMORY FEEDBACK:
- Loss: {loss}
- Grad Norm: {grad_norm}

PERFORMANCE METRICS:
- Average Loss: {avg_loss:.4f}
- Average Grad Norm: {avg_grad_norm:.4f}
- Performance Trend: {trend_status}
- Sample Count: {sample_count}
- Standard Deviation (Loss): {std_dev_loss:.4f}
- System Confidence: {confidence_level}

RECENT HISTORY SUMMARY:
{history_summary}

DECISION BLOCK:""" # LLM completes from here

    def __init__(self, 
                 mode="llmstudio", 
                 llama_endpoint="http://host.docker.internal:1234/v1/chat/completions", 
                 llama_model="bartowski/llama-3.2-1b-instruct",  # Real-time guidance model
                 qwen_model="qwen_qwq-32b",                      # Async/Dream model
                 timeout=15.0,
                 retry_attempts=2,
                 high_surprise_threshold=0.5,
                 low_surprise_threshold=0.1):
        """
        Initialize the LLM router.
        
        Args:
            mode: Operation mode ('llmstudio', 'disabled')
            llama_endpoint: URL for the LM Studio API
            llama_model: Model identifier for real-time guidance
            qwen_model: Model identifier for async/dream tasks
            timeout: Timeout in seconds for API requests
            retry_attempts: Number of retry attempts for failed requests
            high_surprise_threshold: Threshold for high surprise in performance metrics
            low_surprise_threshold: Threshold for low surprise in performance metrics
        """
        self.mode = mode
        
        # Override endpoint with environment variable if available
        env_endpoint = os.environ.get("LLM_STUDIO_ENDPOINT")
        if env_endpoint:
            self.llama_endpoint = env_endpoint
            logger.info(f"Using LLM endpoint from environment: {env_endpoint}")
        else:
            self.llama_endpoint = llama_endpoint
            logger.info(f"Using default LLM endpoint: {llama_endpoint} (No environment variable found)")
            
        # Debug: Print all environment variables to help diagnose issues
        logger.info("Environment variables:")
        for key, value in os.environ.items():
            if "ENDPOINT" in key or "LLM" in key:
                logger.info(f"  {key}: {value}")
                
        self.llama_model = llama_model
        self.qwen_model = qwen_model
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.high_surprise_threshold = high_surprise_threshold
        self.low_surprise_threshold = low_surprise_threshold
        self.session = None
        logger.info(f"MemoryLLMRouter initialized in '{mode}' mode.")
        logger.info(f" - Guidance Model: '{self.llama_model}' at '{self.llama_endpoint}'")
        logger.info(f" - Async Model: '{self.qwen_model}'")
        logger.info(f" - Using thresholds H={high_surprise_threshold}, L={low_surprise_threshold} for prompt.")

    async def _get_session(self):
        """Get or create an aiohttp client session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info(f"Created new aiohttp session with LLM endpoint: {self.llama_endpoint}")
        return self.session

    async def close_session(self):
        """Close the aiohttp client session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def request_llama_guidance(self, 
                                  user_input: str, 
                                  nm_performance: Dict, 
                                  metadata: Dict, 
                                  current_variant: str,
                                  history_summary: str = "[No history available]") -> Dict[str, Any]:
        """
        Request structured guidance from LM Studio based on user input and neural memory feedback.
        
        Args:
            user_input: The user's input text
            nm_performance: Feedback from the neural memory module including loss, grad_norm, and performance metrics
            metadata: Additional context metadata (task_type, emotion, etc.)
            current_variant: The currently active variant (NONE, MAC, MAL, MAG)
            history_summary: Summary of recent memory entries (default placeholder)
            
        Returns:
            Dictionary with structured advice or error info
        """
        if self.mode != "llmstudio":
            logger.warning("LLM Router not in llmstudio mode, skipping guidance request.")
            return self._get_default_llm_guidance()

        try:
            session = await self._get_session()
            
            # Format the prompt with all available information
            prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
                user_input=user_input[:1000],  # Limit input length
                loss=nm_performance.get('loss', 0.0),
                grad_norm=nm_performance.get('grad_norm', 0.0),
                task_type=metadata.get('task_type', 'unknown'),
                emotion=metadata.get('user_emotion', 'neutral'),
                current_variant=current_variant,
                history_summary=history_summary,
                high_surprise_threshold=self.high_surprise_threshold,
                low_surprise_threshold=self.low_surprise_threshold,
                avg_loss=nm_performance.get('avg_loss', 0.0),
                avg_grad_norm=nm_performance.get('avg_grad_norm', 0.0),
                trend_status=nm_performance.get('trend_status', 'unknown'),
                sample_count=nm_performance.get('sample_count', 0),
                std_dev_loss=nm_performance.get('std_dev_loss', 0.0),
                confidence_level=nm_performance.get('confidence_level', 'unknown')
            )

            payload = {
                "model": self.llama_model,
                "messages": [
                    {"role": "system", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for deterministic responses
                "response_format": {
                    "type": "json_schema", 
                    "json_schema": {
                        "schema": self.DEFAULT_LLM_SCHEMA["schema"]
                    }
                }
            }
            
            # Prepare for multiple retry attempts
            for attempt in range(self.retry_attempts + 1):
                try:
                    async with session.post(
                        self.llama_endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"LLM API error (status {response.status}): {error_text}")
                            continue  # Try again
                            
                        result = await response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                        
                        try:
                            parsed_content = json.loads(content)
                            # Validate against schema
                            jsonschema.validate(instance=parsed_content, schema=self.DEFAULT_LLM_SCHEMA)
                            
                            # Add internal trace information about performance metrics
                            if "decision_trace" in parsed_content:
                                performance_summary = f"Performance metrics: loss={nm_performance.get('avg_loss', 0.0):.4f}, grad={nm_performance.get('avg_grad_norm', 0.0):.4f}, trend={nm_performance.get('trend_status', 'unknown')}, confidence={nm_performance.get('confidence_level', 'unknown')}"
                                parsed_content["decision_trace"].append(performance_summary)
                            
                            logger.info(f"LLM guidance request successful. Variant hint: {parsed_content.get('variant_hint', 'NONE')}")
                            return parsed_content
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode LLM response: {str(e)}. Content: {content[:100]}...")
                        except jsonschema.exceptions.ValidationError as e:
                            logger.error(f"LLM response failed schema validation: {str(e)}. Content: {content[:100]}...")
                    
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.retry_attempts:
                        logger.warning(f"LLM request failed (attempt {attempt+1}/{self.retry_attempts+1}): {str(e)}. Retrying...")
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"LLM request failed after {self.retry_attempts+1} attempts: {str(e)}")
            
            # If we've exhausted all retries
            return self._get_default_llm_guidance()
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM guidance request: {str(e)}")
            return self._get_default_llm_guidance()

    def _get_default_llm_guidance(self) -> Dict[str, Any]:
        """
        Returns default guidance when LLM call fails or is disabled.
        
        Returns:
            Dictionary with default advice values
        """
        logger.warning("Returning default LLM advice.")
        
        return {
            "store": True,
            "metadata_tags": ["llm_guidance_failed"],
            "boost_score_mod": 0.0,
            "variant_hint": self.DEFAULT_LLM_SCHEMA["schema"]["properties"]["variant_hint"]["enum"][0], # Default to NONE
            "attention_focus": self.DEFAULT_LLM_SCHEMA["schema"]["properties"]["attention_focus"]["enum"][3], # Default to 'broad'
            "notes": "LLM Guidance Error: Default advice used",
            "decision_trace": ["Using default advice due to LLM failure"]
        }

    async def __aenter__(self):
        """Async context manager enter"""
        await self._get_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
        
    def __del__(self):
        """Destructor to ensure session cleanup"""
        # Create a new event loop if necessary to close the session
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close_session())
                else:
                    loop.run_until_complete(self.close_session())
            except Exception as e:
                logger.warning(f"Failed to close aiohttp session during cleanup: {e}")

    # Helper method to summarize recent history for context - will be implemented in Phase 5.5
    def summarize_recent_history(self, history_items, max_length=500):
        """Create a concise summary of recent history entries for LLM context.
        
        This is a placeholder for Phase 5.5 when we integrate the async memory summarizer.
        In this version, we just concatenate recent entries with minimal formatting.
        
        Args:
            history_items: List of recent history items from sequence_context_manager
            max_length: Maximum length of the summary
            
        Returns:
            String summary of recent history
        """
        if not history_items or not isinstance(history_items, list):
            return "[No history available]"
            
        # Simple concatenation of recent entries (up to 5)
        entries = history_items[-5:] if len(history_items) > 5 else history_items
        summary_parts = []
        
        for idx, entry in enumerate(reversed(entries)):
            # Extract content from entry, fall back to empty string if not found
            content = entry.get("content", "") or ""
            ts = entry.get("timestamp", "unknown time")
            
            # Add entry to summary parts
            if content:
                # Truncate content if too long
                if len(content) > 100:
                    content = content[:97] + "..."
                summary_parts.append(f"[{idx+1}] {content}")
                
        # Join parts and truncate if necessary
        summary = "\n".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary if summary else "[No significant history available]"
