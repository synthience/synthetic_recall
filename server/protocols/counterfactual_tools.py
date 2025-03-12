from typing import Dict, Any, List, Optional, Callable
import logging
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field

from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class CounterfactualToolProvider(ToolProvider):
    """Tool provider specialized for counterfactual reasoning and simulation."""
    
    def __init__(self, self_model=None, world_model=None, memory_system=None, 
                 knowledge_graph=None, parameter_manager=None, model_manager=None):
        super().__init__()
        self.self_model = self_model
        self.world_model = world_model
        self.memory_system = memory_system
        self.knowledge_graph = knowledge_graph
        self.parameter_manager = parameter_manager
        self.model_manager = model_manager
        self.register_counterfactual_tools()
    
    def register_counterfactual_tools(self):
        """Register counterfactual-specific tools."""
        # Timeline branch simulation
        self.register_tool(
            name="simulate_timeline_branch",
            function=self.simulate_timeline_branch,
            description="Simulate an alternative timeline branch based on a changed decision point. "
                      "This helps understand causality and potential outcomes of different choices.",
            parameters={
                "type": "object",
                "properties": {
                    "decision_point": {
                        "type": "string",
                        "description": "The decision point to alter in the timeline",
                    },
                    "alternative_choice": {
                        "type": "string",
                        "description": "The alternative choice to simulate",
                    },
                    "time_horizon": {
                        "type": "integer",
                        "description": "How far into the future to simulate (in steps)",
                        "default": 3
                    }
                },
                "required": ["decision_point", "alternative_choice"],
            }
        )
        
        # Outcome probability estimation
        self.register_tool(
            name="estimate_outcome_probability",
            function=self.estimate_outcome_probability,
            description="Estimate the probability of specific outcomes given a counterfactual scenario. "
                      "Useful for risk assessment and opportunity evaluation.",
            parameters={
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "description": "The counterfactual scenario to evaluate",
                    },
                    "outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of potential outcomes to evaluate"
                    },
                    "context_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional contextual factors to consider"
                    }
                },
                "required": ["scenario", "outcomes"],
            }
        )
        
        # Causal relationship mapping
        self.register_tool(
            name="map_causal_relationships",
            function=self.map_causal_relationships,
            description="Map the causal relationships between events, decisions, and outcomes in a scenario. "
                      "Helps understand complex system dynamics and improve decision modeling.",
            parameters={
                "type": "object",
                "properties": {
                    "central_element": {
                        "type": "string",
                        "description": "The central element (event, decision, outcome) to map relationships for",
                    },
                    "relationship_depth": {
                        "type": "integer",
                        "description": "How many levels of relationships to map",
                        "default": 2
                    },
                    "domain_context": {
                        "type": "string",
                        "description": "Domain context for the causal mapping",
                        "default": "general"
                    }
                },
                "required": ["central_element"],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} counterfactual reasoning tools")
    
    async def simulate_timeline_branch(self, decision_point: str, alternative_choice: str, 
                                time_horizon: int = 3) -> Dict[str, Any]:
        """Simulate an alternative timeline branch based on a changed decision point."""
        try:
            # Simulate timeline branch using LLM if available
            timeline_steps = []
            
            if self.model_manager and hasattr(self.model_manager, "generate_chat_completion"):
                # Convert time_horizon to text for better LLM interpretation
                horizon_text = "short-term" if time_horizon <= 2 else "medium-term" if time_horizon <= 5 else "long-term"
                
                messages = [
                    {"role": "system", "content": "You are a counterfactual timeline simulator for Lucidia's self-evolution system. "
                                            "Your task is to simulate an alternative timeline branch based on a changed decision point, "
                                            "showing how events might unfold differently given the alternative choice."},
                    {"role": "user", "content": f"Decision point: {decision_point}\n\n"
                                            f"Alternative choice: {alternative_choice}\n\n"
                                            f"Time horizon: {horizon_text} ({time_horizon} steps)\n\n"
                                            f"Please simulate the alternative timeline branch, showing each step in the causal chain "
                                            f"and how it differs from what likely happened in the original timeline."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.8,  # Higher temperature for more creative counterfactuals
                    max_tokens=800,   # Reasonable limit for timeline exploration
                    timeout=45        # Allow more time for complex simulation
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    simulation_text = response["choices"][0]["message"].get("content", "")
                    
                    # Try to parse timeline steps
                    import re
                    step_blocks = re.split(r'\n\s*\d+\s*\.\s*|\n\s*Step\s*\d+\s*:|\n\n', simulation_text)
                    step_blocks = [block.strip() for block in step_blocks if block.strip()]
                    
                    for i, block in enumerate(step_blocks[:time_horizon]):  # Limit to requested horizon
                        timeline_steps.append({
                            "step": i + 1,
                            "description": block,
                            "probability": max(0.9 - (i * 0.15), 0.3)  # Decreasing certainty as we go further
                        })
            
            # If no steps generated or LLM unavailable, create fallback steps
            if not timeline_steps:
                for i in range(time_horizon):
                    timeline_steps.append({
                        "step": i + 1,
                        "description": f"Estimated outcome {i+1} following the alternative choice: {alternative_choice} "
                                     f"instead of the original decision at: {decision_point}",
                        "probability": max(0.8 - (i * 0.2), 0.2)  # Decreasing certainty
                    })
            
            return {
                "status": "success",
                "decision_point": decision_point,
                "alternative_choice": alternative_choice,
                "timeline_steps": timeline_steps,
                "time_horizon": time_horizon,
                "confidence": 0.7  # Overall confidence in simulation
            }
            
        except Exception as e:
            logger.error(f"Error simulating timeline branch: {e}")
            return {
                "status": "error",
                "message": f"Error simulating timeline branch: {str(e)}",
                "timeline_steps": []
            }
    
    async def estimate_outcome_probability(self, scenario: str, outcomes: List[str], 
                                    context_factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Estimate the probability of specific outcomes given a counterfactual scenario."""
        try:
            # Default probability distribution method
            estimated_probabilities = {}
            
            # Try to generate probabilities with LLM if available
            if self.model_manager and hasattr(self.model_manager, "generate_chat_completion"):
                context_text = "\n- " + "\n- ".join(context_factors) if context_factors else ""
                
                # Construct the messages with explicit strings to avoid any formatting issues
                system_content = "You are a probability estimator for Lucidia's counterfactual reasoning system. Your task is to estimate the likelihood of different outcomes given a counterfactual scenario and relevant context factors. Express probabilities as decimals between 0 and 1."
                
                # Build the user content step by step
                user_content_parts = [
                    f"Scenario: {scenario}\n\n",
                    f"Possible outcomes:\n"
                ]
                
                # Add each outcome on a separate line
                for outcome in outcomes:
                    user_content_parts.append(f"- {outcome}\n")
                
                # Add context factors and instructions
                user_content_parts.append(f"\nContext factors:{context_text}\n\n")
                user_content_parts.append("For each outcome, please estimate its probability (0-1) and provide a brief justification.")
                
                # Join all parts to create the final user content
                user_content = "".join(user_content_parts)
                
                # Create the messages list
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.4,     # Lower temperature for more consistent probability estimates
                    max_tokens=750,      # Enough for probability analysis
                    top_p=0.9,          # Focus on more likely tokens
                    timeout=40           # Reasonable timeout for analysis
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    probability_text = response["choices"][0]["message"].get("content", "")
                    
                    # Parse probabilities from the text
                    for outcome in outcomes:
                        # Try to match patterns like "Outcome: 0.7" or "Probability: 0.7" near the outcome text
                        import re
                        outcome_pattern = re.escape(outcome)
                        probability_match = re.search(
                            rf"{outcome_pattern}.*?(?:probability|likelihood):?\s*(0\.\d+|\d\.\d+|\d+%)|" + 
                            rf"probability|likelihood.*?{outcome_pattern}:?\s*(0\.\d+|\d\.\d+|\d+%)", 
                            probability_text, re.IGNORECASE
                        )
                        
                        if probability_match:
                            prob_str = probability_match.group(1) or probability_match.group(2)
                            if '%' in prob_str:
                                probability = float(prob_str.strip('%')) / 100
                            else:
                                probability = float(prob_str)
                        else:
                            # If no match, use heuristic
                            probability = 1.0 / len(outcomes)  # Equal distribution
                        
                        # Extract justification if available
                        justification_match = re.search(
                            rf"{outcome_pattern}.*?(because.*?)(?:\n|$)", 
                            probability_text, re.IGNORECASE
                        )
                        justification = justification_match.group(1).strip() if justification_match else "Based on scenario analysis"
                        
                        estimated_probabilities[outcome] = {
                            "probability": probability,
                            "justification": justification
                        }
            
            # If no probabilities were estimated, create default distribution
            if not estimated_probabilities:
                # Simple equal distribution
                equal_prob = 1.0 / len(outcomes)
                for outcome in outcomes:
                    estimated_probabilities[outcome] = {
                        "probability": equal_prob,
                        "justification": "Equal probability distribution (fallback method)"
                    }
            
            return {
                "status": "success",
                "scenario": scenario,
                "estimated_probabilities": estimated_probabilities,
                "context_factors": context_factors or [],
                "confidence": 0.65  # Overall confidence in estimates
            }
            
        except Exception as e:
            logger.error(f"Error estimating outcome probabilities: {e}")
            # Simple equal distribution as fallback
            equal_prob = 1.0 / len(outcomes)
            probabilities = {outcome: {"probability": equal_prob, "justification": "Equal distribution (error fallback)"} 
                            for outcome in outcomes}
            return {
                "status": "partial",
                "message": f"Error estimating probabilities: {str(e)}",
                "estimated_probabilities": probabilities
            }
    
    async def map_causal_relationships(self, central_element: str, relationship_depth: int = 2, 
                                domain_context: str = "general") -> Dict[str, Any]:
        """Map the causal relationships between events, decisions, and outcomes."""
        try:
            relationships = []
            
            # Generate causal map with LLM if available
            if self.model_manager and hasattr(self.model_manager, "generate_chat_completion"):
                messages = [
                    {"role": "system", "content": "You are a causal relationship mapper for Lucidia's counterfactual reasoning system. "
                                            "Your task is to identify and map causal relationships between elements (events, "
                                            "decisions, outcomes) and represent them in a structured format."},
                    {"role": "user", "content": f"Central element: {central_element}\n\n"
                                            f"Relationship depth: {relationship_depth}\n\n"
                                            f"Domain context: {domain_context}\n\n"
                                            f"Please map the causal relationships, showing causes and effects at each level of depth. "
                                            f"For each relationship, indicate direction (cause->effect), strength (0-1), and uncertainty."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.5,     # Balanced temperature for creativity vs. consistency
                    max_tokens=1000,     # Generous limit for complex causal mapping
                    frequency_penalty=0.2,  # Encourage variety in relationships
                    timeout=50           # Allow extra time for complex relationship mapping
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    causal_text = response["choices"][0]["message"].get("content", "")
                    
                    # Try to parse causal relationships from text
                    import re
                    # Look for patterns like "A -> B (0.8)" or "Cause: A, Effect: B, Strength: 0.8"
                    relationship_matches = re.finditer(
                        r"([\w\s]+)\s*(?:->|causes|leads to)\s*([\w\s]+)(?:\s*\(?(0\.\d+|\d\.\d+|\d+%)\)?)?|" +
                        r"Cause:\s*([\w\s]+)\s*,\s*Effect:\s*([\w\s]+)(?:\s*,\s*Strength:\s*(0\.\d+|\d\.\d+|\d+%))?\s*",
                        causal_text, re.IGNORECASE
                    )
                    
                    for match in relationship_matches:
                        # Extract cause, effect, and strength from different match patterns
                        if match.group(1) and match.group(2):  # A -> B pattern
                            cause = match.group(1).strip()
                            effect = match.group(2).strip()
                            strength_str = match.group(3) if match.group(3) else "0.7"
                        else:  # Cause: A, Effect: B pattern
                            cause = match.group(4).strip()
                            effect = match.group(5).strip()
                            strength_str = match.group(6) if match.group(6) else "0.7"
                        
                        # Convert strength to float
                        if '%' in strength_str:
                            strength = float(strength_str.strip('%')) / 100
                        else:
                            strength = float(strength_str)
                        
                        relationships.append({
                            "cause": cause,
                            "effect": effect,
                            "strength": strength,
                            "type": "direct" if cause == central_element or effect == central_element else "indirect"
                        })
            
            # If no relationships parsed, create fallback
            if not relationships:
                # Generate some basic cause-effect pairs
                causes = [f"Cause of {central_element} #{i+1}" for i in range(2)]
                effects = [f"Effect of {central_element} #{i+1}" for i in range(2)]
                
                for i, cause in enumerate(causes):
                    relationships.append({
                        "cause": cause,
                        "effect": central_element,
                        "strength": 0.8 - (i * 0.1),
                        "type": "direct"
                    })
                
                for i, effect in enumerate(effects):
                    relationships.append({
                        "cause": central_element,
                        "effect": effect,
                        "strength": 0.8 - (i * 0.1),
                        "type": "direct"
                    })
            
            return {
                "status": "success",
                "central_element": central_element,
                "relationships": relationships,
                "depth": relationship_depth,
                "domain_context": domain_context,
                "confidence": 0.7  # Overall confidence in mapping
            }
            
        except Exception as e:
            logger.error(f"Error mapping causal relationships: {e}")
            return {
                "status": "error",
                "message": f"Error mapping causal relationships: {str(e)}",
                "relationships": []
            }
