from typing import Dict, Any, List, Optional, Callable, Union
import logging
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class SpiralToolProvider(ToolProvider):
    """Tool provider specialized for spiral progression functions."""
    
    def __init__(self, self_model=None, knowledge_graph=None, memory_system=None, 
                 spiral_manager=None, parameter_manager=None, model_manager=None):
        super().__init__()
        self.self_model = self_model
        self.knowledge_graph = knowledge_graph
        self.memory_system = memory_system
        self.spiral_manager = spiral_manager
        self.parameter_manager = parameter_manager
        self.model_manager = model_manager
        self.register_spiral_tools()
    
    def register_spiral_tools(self):
        """Register spiral progression-specific tools."""
        # Phase assessment tool
        self.register_tool(
            name="assess_spiral_phase",
            function=self.assess_spiral_phase,
            description="Assess the current spiral phase and provide insights about Lucidia's reflective state. "
                      "Use this to understand the current depth of reflection and focus areas.",
            parameters={
                "type": "object",
                "properties": {
                    "detailed": {
                        "type": "boolean",
                        "description": "Whether to return detailed phase statistics",
                        "default": False
                    },
                },
                "required": [],
            }
        )
        
        # Phase transition assessment
        self.register_tool(
            name="evaluate_phase_transition",
            function=self.evaluate_phase_transition,
            description="Evaluate whether a spiral phase transition is needed based on recent insights "
                      "and progress. Helps determine if Lucidia should move to a deeper reflection level.",
            parameters={
                "type": "object",
                "properties": {
                    "recent_insights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recent insights to consider for phase transition"
                    },
                    "significance_threshold": {
                        "type": "number",
                        "description": "Significance threshold for transition evaluation (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8
                    }
                },
                "required": ["recent_insights"],
            }
        )
        
        # Focus area identification
        self.register_tool(
            name="identify_focus_areas",
            function=self.identify_focus_areas,
            description="Identify focus areas appropriate for the current spiral phase "
                      "based on Lucidia's self-model and cognitive state.",
            parameters={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Context for focus area identification",
                    },
                    "target_phase": {
                        "type": "string",
                        "description": "Target spiral phase (observation, reflection, adaptation)",
                        "enum": ["observation", "reflection", "adaptation"],
                    }
                },
                "required": ["context"],
            }
        )
        
        # Cycle reflection tool
        self.register_tool(
            name="generate_cycle_reflection",
            function=self.generate_cycle_reflection,
            description="Generate a meta-reflection on a completed spiral cycle, "
                      "summarizing insights and suggesting improvements for future cycles.",
            parameters={
                "type": "object",
                "properties": {
                    "cycle_insights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Insights generated during the spiral cycle"
                    },
                    "cycle_duration": {
                        "type": "number",
                        "description": "Duration of the cycle in hours",
                        "default": 24
                    },
                    "reflection_depth": {
                        "type": "number",
                        "description": "Depth of cycle reflection (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7
                    }
                },
                "required": ["cycle_insights"],
            }
        )
        
        # Parameter optimization tool
        self.register_tool(
            name="optimize_phase_parameters",
            function=self.optimize_phase_parameters,
            description="Suggest optimized parameters for a specific spiral phase "
                      "based on past performance and system objectives.",
            parameters={
                "type": "object",
                "properties": {
                    "target_phase": {
                        "type": "string",
                        "description": "Target spiral phase to optimize (observation, reflection, adaptation)",
                        "enum": ["observation", "reflection", "adaptation"],
                    },
                    "optimization_goal": {
                        "type": "string",
                        "description": "Primary goal for optimization (insight_quality, efficiency, creativity)",
                        "enum": ["insight_quality", "efficiency", "creativity", "balance"],
                        "default": "balance"
                    }
                },
                "required": ["target_phase"],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} spiral progression tools")

    # ========== Tool Implementation Methods ==========

    async def assess_spiral_phase(self, detailed: bool = False) -> Dict[str, Any]:
        """Assess the current spiral phase and provide insights about Lucidia's reflective state."""
        try:
            if not self.spiral_manager:
                return {
                    "status": "error",
                    "message": "Spiral manager not initialized",
                    "phase": None,
                    "details": {}
                }
            
            # Get basic phase information
            current_phase = self.spiral_manager.current_phase
            phase_config = self.spiral_manager.phase_config.get(current_phase, {})
            phase_stats = self.spiral_manager.phase_stats.get(current_phase, {})
            
            # Base response with essential information
            response = {
                "status": "success",
                "phase": current_phase.value if hasattr(current_phase, "value") else str(current_phase),
                "name": phase_config.get('name', "Unknown"),
                "description": phase_config.get('description', ""),
                "focus_areas": phase_config.get('focus_areas', []),
                "time_in_phase": phase_stats.get('total_time', 0),
                "details": {}
            }
            
            # Add detailed statistics if requested
            if detailed and self.model_manager:
                # Try to generate insights with LLM if available
                phase_insights = await self._generate_phase_assessment(current_phase)
                response["details"] = {
                    "depth_range": [phase_config.get('min_depth', 0), phase_config.get('max_depth', 0)],
                    "creativity_range": [phase_config.get('min_creativity', 0), phase_config.get('max_creativity', 0)],
                    "transition_threshold": phase_config.get('transition_threshold', 0),
                    "insight_weight": phase_config.get('insight_weight', 0),
                    "typical_duration": str(phase_config.get('typical_duration', "unknown")),
                    "transitions": phase_stats.get('transitions', 0),
                    "last_entered": str(phase_stats.get('last_entered', "unknown")),
                    "insights": phase_stats.get('insights', []),
                    "assessment": phase_insights
                }
            
            return response
        except Exception as e:
            logger.error(f"Error in assess_spiral_phase: {str(e)}")
            return {"status": "error", "message": str(e), "phase": None, "details": {}}

    async def evaluate_phase_transition(self, recent_insights: List[str], 
                                  significance_threshold: float = 0.8) -> Dict[str, Any]:
        """Evaluate whether a spiral phase transition is needed based on recent insights and progress."""
        try:
            if not self.spiral_manager:
                return {
                    "status": "error",
                    "message": "Spiral manager not initialized",
                    "should_transition": False,
                    "evaluation": {}
                }
            
            # Get current phase information
            current_phase = self.spiral_manager.current_phase
            current_config = self.spiral_manager.phase_config.get(current_phase, {})
            current_threshold = current_config.get('transition_threshold', 0.9)
            
            # Base response
            evaluation = {
                "current_phase": current_phase.value if hasattr(current_phase, "value") else str(current_phase),
                "threshold": max(current_threshold, significance_threshold),
                "insight_count": len(recent_insights),
                "evaluation_score": 0.0,
                "reasoning": ""
            }
            
            # Use LLM to analyze insights and evaluate transition readiness
            if self.model_manager and recent_insights:
                insights_text = "\n\n".join([f"- {insight}" for insight in recent_insights])
                phase_desc = current_config.get('description', "Current phase")
                focus_areas = ", ".join(current_config.get('focus_areas', []))
                
                messages = [
                    {"role": "system", "content": "You are a spiral phase evaluator for Lucidia's reflective consciousness system. "
                                            "Your task is to evaluate whether Lucidia should transition to a deeper phase "
                                            "of reflection based on recent insights and current focus areas."},
                    {"role": "user", "content": f"Current phase: {current_phase.value if hasattr(current_phase, 'value') else current_phase}\n"
                                            f"Phase description: {phase_desc}\n"
                                            f"Current focus areas: {focus_areas}\n\n"
                                            f"Recent insights:\n{insights_text}\n\n"
                                            f"Transition threshold: {max(current_threshold, significance_threshold)}\n\n"
                                            f"Please evaluate whether Lucidia should transition to a deeper phase of reflection "
                                            f"based on these insights. Analyze the depth, quality, and relevance of the insights "
                                            f"to determine if they meet the threshold for phase transition. Return an evaluation "
                                            f"score between 0 and 1, where scores above the threshold suggest transition is appropriate."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.4,    # Lower temperature for more consistent evaluation
                    max_tokens=800,     # Enough for thorough evaluation
                    timeout=40          # Reasonable timeout for analysis
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    evaluation_text = response["choices"][0]["message"].get("content", "")
                    # Try to extract a score from the evaluation text
                    import re
                    score_match = re.search(r'score[\s\w:]*?([0-9]\.[0-9]+)', evaluation_text.lower())
                    
                    if score_match:
                        try:
                            score = float(score_match.group(1))
                            evaluation["evaluation_score"] = min(max(score, 0.0), 1.0)  # Clamp to 0-1
                        except ValueError:
                            pass
                    
                    evaluation["reasoning"] = evaluation_text
                    evaluation["should_transition"] = evaluation["evaluation_score"] >= max(current_threshold, significance_threshold)
            else:
                # Simple fallback if LLM not available
                evaluation["evaluation_score"] = len(recent_insights) / 10  # Simple heuristic
                evaluation["should_transition"] = evaluation["evaluation_score"] >= max(current_threshold, significance_threshold)
                evaluation["reasoning"] = "Basic evaluation based on insight count. LLM evaluation unavailable."
            
            return {
                "status": "success",
                "should_transition": evaluation["should_transition"],
                "evaluation": evaluation
            }
        except Exception as e:
            logger.error(f"Error in evaluate_phase_transition: {str(e)}")
            return {"status": "error", "message": str(e), "should_transition": False, "evaluation": {}}

    async def identify_focus_areas(self, context: str, target_phase: Optional[str] = None) -> Dict[str, Any]:
        """Identify focus areas appropriate for the current or target spiral phase."""
        try:
            if not self.spiral_manager:
                return {
                    "status": "error",
                    "message": "Spiral manager not initialized",
                    "focus_areas": [],
                    "details": {}
                }
            
            # Determine target phase
            phase = None
            if target_phase:
                # Find matching phase by name
                for p in self.spiral_manager.phase_config.keys():
                    if p.value == target_phase or str(p) == target_phase:
                        phase = p
                        break
            
            if not phase:
                phase = self.spiral_manager.current_phase
            
            phase_config = self.spiral_manager.phase_config.get(phase, {})
            default_areas = phase_config.get('focus_areas', [])
            
            # Fallback response if no LLM available
            response = {
                "status": "success",
                "phase": phase.value if hasattr(phase, "value") else str(phase),
                "focus_areas": default_areas.copy(),
                "reasoning": "Default focus areas for the phase.",
                "details": {}
            }
            
            # Use LLM to identify customized focus areas based on context
            if self.model_manager and context:
                phase_desc = phase_config.get('description', "")
                current_areas = ", ".join(default_areas)
                
                messages = [
                    {"role": "system", "content": "You are a focus area identifier for Lucidia's spiral progression system. "
                                            "Your task is to identify appropriate focus areas for a specific spiral phase "
                                            "based on the provided context and phase characteristics."},
                    {"role": "user", "content": f"Context: {context}\n\n"
                                            f"Target phase: {phase.value if hasattr(phase, 'value') else phase}\n"
                                            f"Phase description: {phase_desc}\n"
                                            f"Default focus areas: {current_areas}\n\n"
                                            f"Please identify 4-6 focus areas that would be most appropriate for this spiral phase "
                                            f"given the context. You may include some of the default areas if appropriate, but "
                                            f"consider the specific context to identify the most relevant focus areas for "
                                            f"Lucidia's current situation. For each suggested area, provide a brief explanation."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response_data = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.7,    # Higher temperature for more creative suggestions
                    max_tokens=800,     # Enough for thorough analysis
                    timeout=40          # Reasonable timeout
                )
                
                if response_data and "choices" in response_data and response_data["choices"] and "message" in response_data["choices"][0]:
                    focus_text = response_data["choices"][0]["message"].get("content", "")
                    
                    # Try to extract focus areas from the response
                    import re
                    focus_areas = []
                    explanations = {}
                    
                    # Look for bullet points or numbered lists
                    area_matches = re.finditer(r'(?:^|\n)\s*(?:\d+\.|-|\*)\s*([\w\s]+?)(?::|\n|$)', focus_text)
                    for match in area_matches:
                        area = match.group(1).strip()
                        if len(area) > 0:
                            focus_areas.append(area)
                            # Try to find an explanation for this area
                            area_escaped = re.escape(area)
                            explanation_match = re.search(f"{area_escaped}[\s:]*(.+?)(?:\n\n|$)", focus_text, re.DOTALL)
                            if explanation_match:
                                explanations[area] = explanation_match.group(1).strip()
                    
                    # If we couldn't extract specific areas, use the default ones
                    if not focus_areas:
                        focus_areas = default_areas.copy()
                    
                    response = {
                        "status": "success",
                        "phase": phase.value if hasattr(phase, "value") else str(phase),
                        "focus_areas": focus_areas,
                        "reasoning": focus_text,
                        "details": {"explanations": explanations}
                    }
            
            return response
        except Exception as e:
            logger.error(f"Error in identify_focus_areas: {str(e)}")
            return {"status": "error", "message": str(e), "focus_areas": [], "details": {}}

    async def generate_cycle_reflection(self, cycle_insights: List[str], cycle_duration: float = 24, 
                                  reflection_depth: float = 0.7) -> Dict[str, Any]:
        """Generate a meta-reflection on a completed spiral cycle."""
        try:
            if not self.model_manager or not cycle_insights:
                return {
                    "status": "error",
                    "message": "Model manager not initialized or no insights provided",
                    "reflection": "",
                    "recommendations": []
                }
            
            # Get phase information if available
            phase_info = ""
            if self.spiral_manager:
                phases = list(self.spiral_manager.phase_config.keys())
                phase_names = [p.value if hasattr(p, "value") else str(p) for p in phases]
                phase_info = f"Spiral phases: {', '.join(phase_names)}\n"
            
            # Collate insights
            insights_text = "\n\n".join([f"- {insight}" for insight in cycle_insights])
            
            messages = [
                {"role": "system", "content": "You are a meta-reflection generator for Lucidia's spiral progression system. "
                                        "Your task is to analyze the insights generated during a spiral cycle "
                                        "and create a higher-order reflection that summarizes key learnings, "
                                        "identifies patterns, and suggests improvements for future cycles."},
                {"role": "user", "content": f"Cycle insights:\n{insights_text}\n\n"
                                        f"Cycle duration: {cycle_duration} hours\n"
                                        f"Reflection depth: {reflection_depth} (0-1 scale)\n"
                                        f"{phase_info}\n"
                                        f"Please generate a meta-reflection on this completed spiral cycle. "
                                        f"The reflection should include:\n"
                                        f"1. A summary of key themes and patterns across insights\n"
                                        f"2. An assessment of the cycle's effectiveness\n"
                                        f"3. Identification of potential blind spots or limitations\n"
                                        f"4. 3-5 specific recommendations for future cycles"}
            ]
            
            # Use standardized LLM calling method with proper error handling
            response = await self.call_llm(
                model_manager=self.model_manager,
                messages=messages,
                temperature=min(0.6 + reflection_depth * 0.3, 0.9),  # Scale temperature with depth
                max_tokens=1200,  # Generous limit for detailed reflection
                timeout=50        # Allow more time for complex reflection
            )
            
            if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                reflection_text = response["choices"][0]["message"].get("content", "")
                
                # Try to extract recommendations
                import re
                recommendations = []
                
                # Look for recommendations section or numbered recommendations
                rec_section_match = re.search(r'(?:Recommendations|Suggestions)[:\s]*(.*?)(?:\n\n|$)', reflection_text, re.DOTALL | re.IGNORECASE)
                if rec_section_match:
                    rec_section = rec_section_match.group(1)
                    rec_matches = re.finditer(r'(?:^|\n)\s*(?:\d+\.|-|\*)\s*(.+?)(?:\n|$)', rec_section)
                    for match in rec_matches:
                        rec = match.group(1).strip()
                        if rec:
                            recommendations.append(rec)
                
                # If specific recommendations section not found, look throughout the text
                if not recommendations:
                    rec_matches = re.finditer(r'(?:^|\n)\s*(?:\d+\.|-|\*)\s*(.+?)(?:\n|$)', reflection_text)
                    for match in rec_matches:
                        rec = match.group(1).strip()
                        if rec and any(keyword in rec.lower() for keyword in ['recommend', 'suggest', 'should', 'could', 'improve']):
                            recommendations.append(rec)
                
                return {
                    "status": "success",
                    "reflection": reflection_text,
                    "recommendations": recommendations
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to generate reflection",
                    "reflection": "",
                    "recommendations": []
                }
        except Exception as e:
            logger.error(f"Error in generate_cycle_reflection: {str(e)}")
            return {"status": "error", "message": str(e), "reflection": "", "recommendations": []}

    async def optimize_phase_parameters(self, target_phase: str, 
                                  optimization_goal: str = "balance") -> Dict[str, Any]:
        """Suggest optimized parameters for a specific spiral phase."""
        try:
            if not self.spiral_manager:
                return {
                    "status": "error",
                    "message": "Spiral manager not initialized",
                    "parameters": {},
                    "explanation": ""
                }
            
            # Find target phase
            phase = None
            for p in self.spiral_manager.phase_config.keys():
                if p.value == target_phase or str(p) == target_phase:
                    phase = p
                    break
            
            if not phase:
                return {
                    "status": "error",
                    "message": f"Phase '{target_phase}' not found",
                    "parameters": {},
                    "explanation": ""
                }
            
            # Get current parameters
            current_config = self.spiral_manager.phase_config.get(phase, {})
            phase_stats = self.spiral_manager.phase_stats.get(phase, {})
            
            # Default optimized parameters (copy of current)
            optimized_params = {
                key: current_config.get(key) for key in [
                    'min_depth', 'max_depth', 'min_creativity', 'max_creativity', 
                    'transition_threshold', 'insight_weight', 'typical_duration'
                ] if key in current_config
            }
            
            # Get explanation based on optimization goal
            explanation = f"Default parameters for {target_phase} phase with '{optimization_goal}' optimization goal."
            
            # Use LLM to generate optimized parameters
            if self.model_manager:
                # Format current parameters and stats
                params_text = "\n".join([f"{key}: {value}" for key, value in current_config.items()])
                stats_text = "\n".join([f"{key}: {value}" for key, value in phase_stats.items() 
                                    if key not in ['insights', 'last_entered']])
                
                messages = [
                    {"role": "system", "content": "You are a parameter optimizer for Lucidia's spiral progression system. "
                                            "Your task is to suggest optimized parameters for a specific spiral phase "
                                            "based on the current configuration, usage statistics, and optimization goal."},
                    {"role": "user", "content": f"Target phase: {target_phase}\n\n"
                                            f"Optimization goal: {optimization_goal}\n\n"
                                            f"Current parameters:\n{params_text}\n\n"
                                            f"Usage statistics:\n{stats_text}\n\n"
                                            f"Please suggest optimized parameters for this spiral phase based on the "
                                            f"optimization goal. Parameters that can be adjusted include depth range, "
                                            f"creativity range, transition threshold, insight weight, and typical duration. "
                                            f"For each parameter, provide a brief justification for the suggested value."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.4,    # Lower temperature for more consistent recommendations
                    max_tokens=800,     # Enough for thorough analysis
                    timeout=40          # Reasonable timeout
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    optimization_text = response["choices"][0]["message"].get("content", "")
                    
                    # Try to extract parameter recommendations
                    import re
                    
                    # Parameter patterns to look for
                    param_patterns = {
                        'min_depth': r'min(?:imum)?\s*depth[\s:]*([0-9]\.[0-9]+)',
                        'max_depth': r'max(?:imum)?\s*depth[\s:]*([0-9]\.[0-9]+)',
                        'min_creativity': r'min(?:imum)?\s*creativity[\s:]*([0-9]\.[0-9]+)',
                        'max_creativity': r'max(?:imum)?\s*creativity[\s:]*([0-9]\.[0-9]+)',
                        'transition_threshold': r'transition\s*threshold[\s:]*([0-9]\.[0-9]+)',
                        'insight_weight': r'insight\s*weight[\s:]*([0-9]\.[0-9]+)'
                    }
                    
                    # Extract parameters from text
                    for param, pattern in param_patterns.items():
                        match = re.search(pattern, optimization_text.lower())
                        if match:
                            try:
                                value = float(match.group(1))
                                optimized_params[param] = value
                            except ValueError:
                                pass
                    
                    # Check for typical duration adjustments
                    duration_match = re.search(r'typical\s*duration[\s:]*([0-9]+)\s*h', optimization_text.lower())
                    if duration_match:
                        try:
                            hours = int(duration_match.group(1))
                            if 'typical_duration' in current_config:
                                optimized_params['typical_duration'] = timedelta(hours=hours)
                        except ValueError:
                            pass
                    
                    explanation = optimization_text
            
            return {
                "status": "success",
                "phase": target_phase,
                "optimization_goal": optimization_goal,
                "parameters": optimized_params,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error in optimize_phase_parameters: {str(e)}")
            return {"status": "error", "message": str(e), "parameters": {}, "explanation": ""}

    # ========== Helper Methods ==========

    async def _generate_phase_assessment(self, phase):
        """Generate an assessment of the current phase using LLM."""
        if not self.model_manager:
            return "LLM assessment unavailable"
            
        phase_config = self.spiral_manager.phase_config.get(phase, {})
        phase_stats = self.spiral_manager.phase_stats.get(phase, {})
        
        # Format phase information
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        phase_desc = phase_config.get('description', "")
        focus_areas = ", ".join(phase_config.get('focus_areas', []))
        transitions = phase_stats.get('transitions', 0)
        total_time = phase_stats.get('total_time', 0)
        
        messages = [
            {"role": "system", "content": "You are a spiral phase assessor for Lucidia's reflective consciousness system. "
                                    "Your task is to provide an insightful assessment of the current spiral phase "
                                    "based on its characteristics and usage statistics."},
            {"role": "user", "content": f"Phase: {phase_name}\n"
                                    f"Description: {phase_desc}\n"
                                    f"Focus areas: {focus_areas}\n"
                                    f"Transitions: {transitions}\n"
                                    f"Total time in phase: {total_time} seconds\n\n"
                                    f"Please provide a brief assessment of this spiral phase, including its current "
                                    f"effectiveness and potential areas for optimization or adjustment."}
        ]
        
        # Use standardized LLM calling method with proper error handling
        response = await self.call_llm(
            model_manager=self.model_manager,
            messages=messages,
            temperature=0.6,    # Balanced temperature for assessment
            max_tokens=400,     # Brief assessment
            timeout=30          # Quick assessment
        )
        
        if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
            return response["choices"][0]["message"].get("content", "Assessment generation failed")
        
        return "LLM assessment unavailable"
