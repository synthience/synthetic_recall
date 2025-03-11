"""
Lucidia's ReflectionEngine

This module implements Lucidia's reflection engine for periodically reviewing and refining
dream reports. The reflection engine analyzes new information, updates confidence levels,
and enhances the dream reports over time to improve Lucidia's metacognitive abilities.
"""

import time
import logging
import asyncio
import uuid
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from uuid import uuid4

from memory.lucidia_memory_system.core.dream_structures import DreamReport, DreamFragment
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph
from memory.lucidia_memory_system.core.memory_entry import MemoryEntry
from memory.lucidia_memory_system.core.integration import MemoryIntegration
from memory.lucidia_memory_system.core.hypersphere_dispatcher import HypersphereDispatcher

# Define prompt templates for the reflection engine
REFLECTION_PROMPT = """
You are analyzing new evidence related to a dream report titled "{report_title}".

Existing report components:

INSIGHTS:
{insights_text}

QUESTIONS:
{questions_text}

HYPOTHESES:
{hypotheses_text}

COUNTERFACTUALS:
{counterfactuals_text}

NEW MEMORIES TO CONSIDER:
{memories_text}

Analyze how these new memories relate to the existing report components. For each component, determine:
1. Whether the new memories provide supporting evidence
2. Whether the new memories provide contradicting evidence
3. How confidence levels should be adjusted based on the new evidence

Provide your analysis in JSON format:
{{"supporting_evidence": [{{
    "content": "Fragment content being supported",
    "evidence": "Description of supporting evidence",
    "strength": 0.7 // Number between 0-1 indicating strength of support
}}],
"contradicting_evidence": [{{
    "content": "Fragment content being contradicted",
    "evidence": "Description of contradicting evidence",
    "strength": 0.6 // Number between 0-1 indicating strength of contradiction
}}],
"confidence_adjustments": [{{
    "content": "Fragment content",
    "old_confidence": 0.7, // Original confidence level
    "new_confidence": 0.8, // Suggested new confidence level
    "reason": "Reason for confidence adjustment"
}}]
}}
"""

ACTION_ITEMS_PROMPT = """
You are analyzing low-confidence components in a dream report titled "{report_title}"
 to generate action items for further investigation.

Low-confidence fragments:
{fragments_text}

Based on these low-confidence fragments, generate 3-5 specific action items that would help improve 
understanding or confidence in these areas. Each action item should be concretely actionable.

Provide your response in JSON format:
{{"action_items": [
    "Action item 1 description",
    "Action item 2 description",
    // Additional action items
]
}}
"""

SELF_ASSESSMENT_PROMPT = """
Generate a brief self-assessment for a dream report titled "{report_title}".

Report statistics:
- Confidence level: {confidence}
- Relevance score: {relevance}
- Number of fragments: {num_fragments}
- Supporting evidence items: {num_supporting_evidence}
- Contradicting evidence items: {num_contradicting_evidence}
- Action items: {num_action_items}
- Refinement count: {refinement_count}/{max_refinements}

In 2-3 sentences, summarize the quality and reliability of this report based on these metrics.
Focus on the report's strengths, limitations, and areas for improvement.
"""

class ReflectionEngine:
    """
    Engine for periodically reviewing and refining dream reports.
    
    The ReflectionEngine analyzes new information, updates confidence levels,
    and enhances dream reports over time to improve Lucidia's metacognitive abilities.
    It serves as a critical component for iterative knowledge refinement.
    """
    
    def __init__(
        self,
        knowledge_graph: LucidiaKnowledgeGraph,
        memory_integration: Optional[MemoryIntegration] = None,
        llm_service = None,
        review_interval: int = 3600,
        config: Optional[Dict[str, Any]] = None,
        hypersphere_dispatcher: Optional[HypersphereDispatcher] = None
    ):
        """
        Initialize the Reflection Engine.
        
        Args:
            knowledge_graph: Reference to Lucidia's knowledge graph
            memory_integration: Optional reference to memory integration layer
            llm_service: Service for language model operations
            review_interval: Interval in seconds between review cycles
            config: Optional configuration dictionary
            hypersphere_dispatcher: Optional dispatcher for efficient embedding operations
        """
        self.logger = logging.getLogger("ReflectionEngine")
        self.logger.info("Initializing Lucidia Reflection Engine")
        
        # Store component references
        self.knowledge_graph = knowledge_graph
        self.memory_integration = memory_integration
        self.llm_service = llm_service
        self.hypersphere_dispatcher = hypersphere_dispatcher
        
        # Configuration
        self.config = config or {}
        self.review_interval = review_interval
        self.running = False
        self.review_task = None
        
        # Review criteria
        self.min_confidence_for_review = self.config.get("min_confidence_for_review", 0.8)
        self.max_reports_per_cycle = self.config.get("max_reports_per_cycle", 5)
        self.prioritize_by = self.config.get("prioritize_by", ["last_reviewed", "confidence", "relevance"])
        
        # Review stats
        self.review_stats = {
            "total_reviews": 0,
            "total_updated": 0,
            "total_refinements": 0,
            "last_review_cycle": None,
            "reports_in_system": 0
        }
        
        # Current model version used for embeddings
        self.default_model_version = self.config.get("default_model_version", "latest")
        
        self.logger.info(f"Reflection Engine initialized with review interval: {review_interval}s")
    
    async def start(self) -> Dict[str, Any]:
        """
        Start the reflection engine's review cycle.
        
        Returns:
            Status information about the started service
        """
        if self.running:
            self.logger.info("Reflection Engine already running")
            return {"status": "already_running"}
        
        self.logger.info("Starting Reflection Engine review cycle")
        self.running = True
        self.review_task = asyncio.create_task(self.review_cycle())
        
        return {
            "status": "started",
            "review_interval": self.review_interval,
            "max_reports_per_cycle": self.max_reports_per_cycle
        }
    
    async def stop(self) -> Dict[str, Any]:
        """
        Stop the reflection engine's review cycle.
        
        Returns:
            Status information about the stopped service
        """
        if not self.running:
            self.logger.info("Reflection Engine already stopped")
            return {"status": "already_stopped"}
        
        self.logger.info("Stopping Reflection Engine")
        self.running = False
        
        if self.review_task:
            self.review_task.cancel()
            try:
                await self.review_task
            except asyncio.CancelledError:
                pass
            self.review_task = None
        
        return {
            "status": "stopped",
            "stats": self.review_stats
        }
    
    async def review_cycle(self) -> None:
        """
        Main review cycle for dream reports.
        
        This method runs continuously while the engine is active,
        selecting reports for review and processing them.
        """
        self.logger.info("Starting dream report review cycle")
        
        while self.running:
            try:
                # Get the current count of reports in the system
                self.review_stats["reports_in_system"] = await self.get_report_count()
                
                # Retrieve reports due for review
                reports_to_review = await self.get_reports_for_review()
                
                if reports_to_review:
                    self.logger.info(f"Found {len(reports_to_review)} reports to review")
                    
                    for report in reports_to_review:
                        if not self.running:
                            break
                        
                        try:
                            # Process each report
                            result = await self.refine_report(report)
                            
                            if result.get("updated", False):
                                self.review_stats["total_updated"] += 1
                            
                            self.review_stats["total_reviews"] += 1
                            
                        except Exception as report_error:
                            self.logger.error(f"Error processing report {report.report_id}: {report_error}")
                    
                    # Update stats
                    self.review_stats["last_review_cycle"] = time.time()
                else:
                    self.logger.info("No reports due for review")
                
                # Sleep until the next review cycle
                await asyncio.sleep(self.review_interval)
                
            except Exception as e:
                self.logger.error(f"Error during review cycle: {e}")
                # Short sleep on error before retrying
                await asyncio.sleep(60)
    
    async def get_report_count(self) -> int:
        """
        Get the total number of dream reports in the system.
        
        Returns:
            Count of dream reports
        """
        try:
            # Query the knowledge graph for reports
            count = await self.knowledge_graph.count_nodes(node_type="dream_report")
            return count
        except Exception as e:
            self.logger.error(f"Error getting report count: {e}")
            return 0
    
    async def get_reports_for_review(self) -> List[DreamReport]:
        """
        Get dream reports that are due for review.
        
        Selection criteria:
        1. Reports that haven't been reviewed yet
        2. Reports that were last reviewed longer than review_interval ago
        3. Reports with low confidence that need refinement
        4. Reports with high relevance
        
        Returns:
            List of dream reports due for review
        """
        try:
            # Define the query criteria
            current_time = time.time()
            review_time_threshold = current_time - self.review_interval
            
            # Build the filter for graph query
            filters = {
                "$or": [
                    {"last_reviewed": None},  # Never reviewed
                    {"last_reviewed": {"$lt": review_time_threshold}}  # Reviewed too long ago
                ]
            }
            
            # Add confidence filter for low confidence reports
            if "confidence" in self.prioritize_by:
                filters["analysis.confidence_level"] = {"$lt": self.min_confidence_for_review}
            
            # Query the knowledge graph for reports matching our criteria
            report_nodes = await self.knowledge_graph.query_nodes(
                node_type="dream_report",
                filters=filters,
                limit=self.max_reports_per_cycle,
                sort_by=self.prioritize_by[0] if self.prioritize_by else None
            )
            
            # Convert node data to DreamReport objects
            reports = []
            for node_data in report_nodes:
                try:
                    report = DreamReport.from_dict(node_data["attributes"])
                    reports.append(report)
                except Exception as node_error:
                    self.logger.error(f"Error converting node to report: {node_error}")
            
            return reports
            
        except Exception as e:
            self.logger.error(f"Error getting reports for review: {e}")
            return []
    
    async def get_new_relevant_memories(self, report: DreamReport) -> List[MemoryEntry]:
        """
        Retrieves new memories relevant to the given dream report.
        
        This method looks for memories that:
        1. Were created after the report was last reviewed
        2. Are semantically related to the content of the report
        
        Args:
            report: The dream report to find relevant memories for
            
        Returns:
            List of relevant memory entries
        """
        relevant_memories = []
        
        try:
            # Determine the time threshold (when the report was last created or reviewed)
            time_threshold = report.last_reviewed or report.created_at
            
            # Collect all fragment IDs from the report
            all_fragment_ids = []
            all_fragment_ids.extend(report.insight_ids)
            all_fragment_ids.extend(report.question_ids)
            all_fragment_ids.extend(report.hypothesis_ids)
            all_fragment_ids.extend(report.counterfactual_ids)
            
            # Get memory IDs for existing participating memories
            known_memory_ids = set(report.participating_memory_ids)
            
            # Get fragment content to use for similarity search
            fragment_contents = []
            fragment_ids = []
            
            for fragment_id in all_fragment_ids:
                fragment_node = await self.knowledge_graph.get_node(fragment_id)
                if not fragment_node:
                    continue
                
                fragment_content = fragment_node.get("attributes", {}).get("content", "")
                if fragment_content:
                    fragment_contents.append(fragment_content)
                    fragment_ids.append(fragment_id)
            
            # Use the hypersphere dispatcher for efficient batch embedding if available
            if self.hypersphere_dispatcher and fragment_contents:
                try:
                    # Get embeddings for all fragments in a single batch
                    fragment_embeddings = []
                    for content in fragment_contents:
                        embedding_result = await self.hypersphere_dispatcher.get_embedding(
                            text=content,
                            model_version=self.default_model_version
                        )
                        if "embedding" in embedding_result:
                            fragment_embeddings.append(embedding_result["embedding"])
                    
                    # Get candidate memories (use memory integration for initial candidates)
                    candidate_memories = []
                    if self.memory_integration:
                        # Find memories created after the time threshold
                        recent_memories = await self.memory_integration.get_memories_by_timerange(
                            start_time=time_threshold,
                            end_time=None,  # up to now
                            limit=100
                        )
                        candidate_memories.extend(recent_memories)
                    
                    # If we have both fragment embeddings and candidate memories
                    if fragment_embeddings and candidate_memories:
                        # Extract memory embeddings and IDs
                        memory_embeddings = []
                        memory_ids = []
                        memory_objects = []
                        
                        for memory in candidate_memories:
                            if hasattr(memory, "embedding") and memory.embedding and memory.id not in known_memory_ids:
                                memory_embeddings.append(memory.embedding)
                                memory_ids.append(memory.id)
                                memory_objects.append(memory)
                        
                        # Perform batch similarity search for each fragment embedding
                        for i, fragment_embedding in enumerate(fragment_embeddings):
                            if not memory_embeddings:  # Skip if no memory embeddings
                                break
                                
                            similarity_results = await self.hypersphere_dispatcher.batch_similarity_search(
                                query_embedding=fragment_embedding,
                                memory_embeddings=memory_embeddings,
                                memory_ids=memory_ids,
                                model_version=self.default_model_version,
                                top_k=10
                            )
                            
                            # Add similar memories to the relevant set
                            for result in similarity_results:
                                if result.get("score", 0) >= 0.7:  # Similarity threshold
                                    memory_idx = memory_ids.index(result.get("memory_id"))
                                    memory = memory_objects[memory_idx]
                                    if memory.id not in known_memory_ids:
                                        relevant_memories.append(memory)
                                        known_memory_ids.add(memory.id)
                                        
                        self.logger.info(f"Found {len(relevant_memories)} relevant memories using hypersphere batch search")
                except Exception as e:
                    self.logger.warning(f"Error in hypersphere batch search: {e}. Falling back to standard search.")
            
            # Fallback to traditional search if hypersphere search failed or isn't available
            if not relevant_memories and self.memory_integration:
                for fragment_id in all_fragment_ids:
                    # Get fragment content to use for semantic search
                    fragment_node = await self.knowledge_graph.get_node(fragment_id)
                    if not fragment_node:
                        continue
                    
                    fragment_content = fragment_node.get("attributes", {}).get("content", "")
                    
                    # Use memory integration to find related memories
                    if fragment_content:
                        # Find memories semantically similar to the fragment content
                        similar_memories = await self.memory_integration.find_similar_memories(
                            text=fragment_content,
                            limit=10,
                            threshold=0.7,
                            created_after=time_threshold
                        )
                        
                        # Add memories that aren't already part of the report
                        for memory in similar_memories:
                            if memory.id not in known_memory_ids:
                                relevant_memories.append(memory)
                                known_memory_ids.add(memory.id)
            
            # Also look for memories directly connected to concepts in the report
            # Get concepts mentioned in the report fragments
            concepts = set()
            for fragment_id in all_fragment_ids:
                # Find concepts connected to this fragment
                connected_concepts = await self.knowledge_graph.get_connected_nodes(
                    fragment_id,
                    edge_types=["references", "mentions", "about"],
                    node_types=["concept", "entity"],
                    direction="outbound"
                )
                concepts.update(connected_concepts)
            
            # For each concept, find recent memories related to it
            for concept in concepts:
                # Find memories connected to this concept
                memory_ids = await self.knowledge_graph.get_connected_nodes(
                    concept,
                    edge_types=["references", "mentions", "about"],
                    node_types=["memory"],
                    direction="inbound"
                )
                
                # Retrieve and filter the memories
                for memory_id in memory_ids:
                    if memory_id in known_memory_ids:
                        continue
                    
                    if self.memory_integration:
                        memory = await self.memory_integration.get_memory_by_id(memory_id)
                        if memory and memory.created_at > time_threshold:
                            relevant_memories.append(memory)
                            known_memory_ids.add(memory_id)
            
            return relevant_memories
            
        except Exception as e:
            self.logger.error(f"Error getting new relevant memories: {e}")
            return []
    
    async def update_report_with_new_evidence(
        self, 
        report: DreamReport, 
        new_memories: List[MemoryEntry]
    ) -> DreamReport:
        """
        Updates the dream report based on newly acquired memories.
        
        This method uses the LLM to analyze the new memories in relation to the
        existing report and update the evidence and confidence accordingly.
        
        Args:
            report: The dream report to update
            new_memories: New relevant memories to incorporate
            
        Returns:
            Updated dream report
        """
        if not new_memories or not self.llm_service:
            return report
        
        try:
            # Get the fragments referenced in the report
            insights = await self._get_fragments_by_ids(report.insight_ids)
            questions = await self._get_fragments_by_ids(report.question_ids)
            hypotheses = await self._get_fragments_by_ids(report.hypothesis_ids)
            counterfactuals = await self._get_fragments_by_ids(report.counterfactual_ids)
            
            # Format the fragments for the prompt
            insights_text = "\n".join([f"- {insight.content}" for insight in insights])
            questions_text = "\n".join([f"- {question.content}" for question in questions])
            hypotheses_text = "\n".join([f"- {hypothesis.content}" for hypothesis in hypotheses])
            counterfactuals_text = "\n".join([f"- {counterfactual.content}" for counterfactual in counterfactuals])
            
            # Format the new memories for the prompt
            memories_text = "\n".join([
                f"- Memory {i+1}: {memory.content} (created: {datetime.fromtimestamp(memory.created_at).strftime('%Y-%m-%d')})"
                for i, memory in enumerate(new_memories)
            ])
            
            # Construct the prompt for the LLM
            prompt = REFLECTION_PROMPT.format(
                report_title=report.title,
                insights_text=insights_text,
                questions_text=questions_text,
                hypotheses_text=hypotheses_text,
                counterfactuals_text=counterfactuals_text,
                memories_text=memories_text
            )
            
            # Send the prompt to the LLM
            analysis_result = await self.llm_service.generate_text(prompt)
            
            # Parse the LLM response to extract evidence and confidence adjustments
            analysis_data = json.loads(analysis_result)
            supporting_evidence = analysis_data.get("supporting_evidence", [])
            contradicting_evidence = analysis_data.get("contradicting_evidence", [])
            confidence_adjustments = analysis_data.get("confidence_adjustments", [])
            
            # Update the report's analysis based on the parsed results
            if supporting_evidence:
                report.analysis["supporting_evidence"].extend(supporting_evidence)
            
            if contradicting_evidence:
                report.analysis["contradicting_evidence"].extend(contradicting_evidence)
            
            # Apply confidence adjustments to the fragments
            if confidence_adjustments:
                for adjustment in confidence_adjustments:
                    # Find the fragment by content and update its confidence
                    fragment_id = self._find_fragment_id_by_content(
                        adjustment["content"], 
                        insights + questions + hypotheses + counterfactuals
                    )
                    if fragment_id:
                        await self._update_fragment_confidence(
                            fragment_id, 
                            adjustment["new_confidence"], 
                            adjustment["reason"]
                        )
            
            # Update the memory IDs in the report
            for memory in new_memories:
                if memory.id not in report.participating_memory_ids:
                    report.participating_memory_ids.append(memory.id)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error updating report with new evidence: {e}")
            return report
    
    async def reassess_report(self, report: DreamReport) -> DreamReport:
        """
        Reassess the dream report's overall analysis based on updated fragments.
        
        This updates confidence levels, relevance scores, and generates new action items.
        
        Args:
            report: The dream report to reassess
            
        Returns:
            Updated dream report
        """
        try:
            # Check if we've reached the maximum number of refinements
            if report.is_at_convergence_limit():
                self.logger.info(f"Report {report.report_id} has reached maximum refinement limit ({report.max_refinements}). Skipping further refinement.")
                return report
            
            # Get all fragments referenced by the report
            all_fragments = []
            all_fragments.extend(await self._get_fragments_by_ids(report.insight_ids))
            all_fragments.extend(await self._get_fragments_by_ids(report.question_ids))
            all_fragments.extend(await self._get_fragments_by_ids(report.hypothesis_ids))
            all_fragments.extend(await self._get_fragments_by_ids(report.counterfactual_ids))
            
            if not all_fragments:
                return report
            
            # Calculate overall confidence based on fragment confidences
            total_confidence = sum(fragment.confidence for fragment in all_fragments)
            avg_confidence = total_confidence / len(all_fragments) if all_fragments else 0.5
            
            # Apply diminishing returns to confidence updates based on refinement count
            # The impact of new evidence decreases as refinement count increases
            if report.refinement_count > 0:
                # Calculate diminishing impact factor (gets smaller with more refinements)
                diminishing_factor = 1.0 / (1.0 + (report.refinement_count * 0.2))
                
                # Apply diminishing factor to confidence delta
                old_confidence = report.analysis.get("confidence_level", 0.5)
                confidence_delta = (avg_confidence - old_confidence) * diminishing_factor
                new_confidence = old_confidence + confidence_delta
            else:
                new_confidence = avg_confidence
            
            # Check if confidence is oscillating (alternating up and down)
            if report.is_confidence_oscillating():
                self.logger.info(f"Detected oscillating confidence pattern in report {report.report_id}. Stabilizing confidence value.")
                # Stabilize by using average of last few values
                recent_confidences = report.confidence_history[-4:] + [new_confidence]
                new_confidence = sum(recent_confidences) / len(recent_confidences)
            
            # Check if the confidence change is significant enough to warrant an update
            if not report.is_confidence_change_significant(new_confidence):
                self.logger.info(f"Confidence change in report {report.report_id} is below threshold. No significant update needed.")
                # Still increment the refinement count and record confidence
                report.record_confidence(new_confidence)
                return report
                
            # Update report confidence
            report.analysis["confidence_level"] = new_confidence
            
            # Record this confidence value and increment refinement count
            report.record_confidence(new_confidence)
            
            # Generate action items based on low-confidence fragments
            low_confidence_fragments = [f for f in all_fragments if f.confidence < 0.6]
            if low_confidence_fragments and self.llm_service:
                # Create prompt for generating action items
                fragments_text = "\n".join([
                    f"- {f.fragment_type.upper()}: {f.content} (confidence: {f.confidence})"
                    for f in low_confidence_fragments
                ])
                
                prompt = ACTION_ITEMS_PROMPT.format(
                    report_title=report.title,
                    fragments_text=fragments_text
                )
                
                # Generate action items using the LLM
                action_items_text = await self.llm_service.generate_text(prompt)
                
                # Parse action items
                action_items_data = json.loads(action_items_text)
                action_items = action_items_data.get("action_items", [])
                
                # Add new action items to the report
                report.analysis["action_items"].extend(action_items)
            
            # Calculate relevance score based on:
            # 1. The number of connections to other concepts
            # 2. The confidence level
            # 3. The recency of evidence
            
            # First, get the number of connections to other concepts
            connection_count = 0
            for fragment_id in report.insight_ids + report.question_ids + report.hypothesis_ids + report.counterfactual_ids:
                try:
                    connections = await self.knowledge_graph.get_edge_count(fragment_id)
                    connection_count += connections
                except Exception:
                    pass
            
            # Calculate relevance factor based on connections
            connection_factor = min(1.0, connection_count / 20)  # Cap at 1.0
            
            # Calculate recency factor
            current_time = time.time()
            oldest_memory_time = current_time
            
            for memory_id in report.participating_memory_ids:
                try:
                    memory_node = await self.knowledge_graph.get_node(memory_id)
                    if memory_node and "attributes" in memory_node:
                        memory_time = memory_node["attributes"].get("created_at", current_time)
                        oldest_memory_time = min(oldest_memory_time, memory_time)
                except Exception:
                    pass
            
            # Calculate age in days
            age_days = (current_time - oldest_memory_time) / (60 * 60 * 24)
            recency_factor = max(0.2, min(1.0, 1.0 - (age_days / 365)))  # Decay over a year
            
            # Combine factors for final relevance score
            relevance_score = (0.4 * new_confidence) + (0.4 * connection_factor) + (0.2 * recency_factor)
            report.analysis["relevance_score"] = relevance_score
            
            # Generate a self-assessment using the LLM
            if self.llm_service:
                # Add refinement information to the prompt
                self_assessment_prompt = SELF_ASSESSMENT_PROMPT.format(
                    report_title=report.title,
                    confidence=report.analysis["confidence_level"],
                    relevance=report.analysis["relevance_score"],
                    num_fragments=len(all_fragments),
                    num_supporting_evidence=len(report.analysis["supporting_evidence"]),
                    num_contradicting_evidence=len(report.analysis["contradicting_evidence"]),
                    num_action_items=len(report.analysis["action_items"]),
                    refinement_count=report.refinement_count,
                    max_refinements=report.max_refinements
                )
                
                report.analysis["self_assessment"] = await self.llm_service.generate_text(self_assessment_prompt)
            
            # Log convergence information
            self.logger.info(
                f"Reassessed report {report.report_id} (refinement {report.refinement_count}/{report.max_refinements}): "
                f"confidence={new_confidence:.2f}, relevance={relevance_score:.2f}"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error reassessing report: {e}")
            return report
    
    async def update_report_in_graph(self, report: DreamReport) -> bool:
        """
        Update the dream report in the knowledge graph.
        
        Args:
            report: The updated dream report
            
        Returns:
            Success status
        """
        try:
            # Update the report node attributes
            node_attributes = {
                "title": report.title,
                "participating_memory_ids": report.participating_memory_ids,
                "insight_ids": report.insight_ids,
                "question_ids": report.question_ids,
                "hypothesis_ids": report.hypothesis_ids,
                "counterfactual_ids": report.counterfactual_ids,
                "analysis": report.analysis,
                "domain": report.domain,
                "created_at": report.created_at,
                "last_reviewed": report.last_reviewed
            }
            
            # Update the node in the knowledge graph
            success = await self.knowledge_graph.update_node(report.report_id, node_attributes)
            
            # Update connections to participating memories
            if success:
                # Add connections to any new participating memories
                for memory_id in report.participating_memory_ids:
                    if await self.knowledge_graph.has_node(memory_id):
                        # Check if connection already exists
                        existing_edge = await self.knowledge_graph.has_edge(report.report_id, memory_id, "based_on")
                        
                        if not existing_edge:
                            await self.knowledge_graph.add_edge(
                                report.report_id,
                                memory_id,
                                edge_type="based_on",
                                attributes={
                                    "strength": 0.8,
                                    "created": time.time()
                                }
                            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating report in graph: {e}")
            return False
    
    async def refine_report(self, report: DreamReport) -> Dict[str, Any]:
        """
        Refine a dream report by incorporating new evidence and updating analyses.
        
        This is the main method that orchestrates the entire refinement process.
        
        Args:
            report: The dream report to refine
            
        Returns:
            Result information dictionary
        """
        self.logger.info(f"Refining report: {report.title} (ID: {report.report_id})")
        
        try:
            # Check if this report has already reached its refinement limit
            if report.is_at_convergence_limit():
                self.logger.info(f"Report {report.report_id} has reached maximum refinement limit ({report.max_refinements}). Skipping refinement.")
                return {
                    "status": "skipped",
                    "report_id": report.report_id,
                    "updated": False,
                    "reason": f"Maximum refinement limit reached ({report.refinement_count}/{report.max_refinements})",
                    "confidence": report.analysis["confidence_level"],
                    "relevance": report.analysis["relevance_score"]
                }
            
            # 1. Retrieve new relevant memories added since last review
            new_memories = await self.get_new_relevant_memories(report)
            self.logger.info(f"Found {len(new_memories)} new relevant memories for report {report.report_id}")
            
            # Skip refinement if there are no new memories and confidence is already high
            if not new_memories and report.analysis.get("confidence_level", 0) > 0.8:
                self.logger.info(f"No new memories found and confidence is already high for report {report.report_id}. Skipping refinement.")
                return {
                    "status": "skipped",
                    "report_id": report.report_id,
                    "updated": False,
                    "reason": "No new memories and high confidence",
                    "confidence": report.analysis["confidence_level"],
                    "relevance": report.analysis["relevance_score"]
                }
            
            # 2. Analyze new memories in relation to report
            if new_memories:
                report = await self.update_report_with_new_evidence(report, new_memories)
                self.logger.info(f"Updated report {report.report_id} with new evidence")
            
            # Check if confidence is oscillating before reassessment
            if report.is_confidence_oscillating():
                self.logger.warning(f"Detected oscillating confidence pattern in report {report.report_id}. Proceeding with caution.")
            
            # 3. Reassess the report's overall analysis
            old_confidence = report.analysis.get("confidence_level", 0)
            report = await self.reassess_report(report)
            new_confidence = report.analysis.get("confidence_level", 0)
            
            # Check if the confidence actually changed significantly
            confidence_change = abs(new_confidence - old_confidence)
            if confidence_change < report.significant_update_threshold:
                self.logger.info(f"Minimal confidence change ({confidence_change:.4f}) for report {report.report_id}")
            
            # 4. Update the last_reviewed timestamp
            report.last_reviewed = time.time()
            
            # 5. Save the updated report to the knowledge graph
            success = await self.update_report_in_graph(report)
            
            if success:
                self.review_stats["total_refinements"] += 1
                self.logger.info(
                    f"Successfully refined report {report.report_id} "
                    f"(refinement {report.refinement_count}/{report.max_refinements})"
                )
                
                return {
                    "status": "success",
                    "report_id": report.report_id,
                    "updated": True,
                    "new_memories_count": len(new_memories),
                    "confidence": report.analysis["confidence_level"],
                    "relevance": report.analysis["relevance_score"],
                    "refinement_count": report.refinement_count,
                    "max_refinements": report.max_refinements,
                    "confidence_change": confidence_change
                }
            else:
                self.logger.error(f"Failed to save refined report {report.report_id} to knowledge graph")
                return {
                    "status": "error",
                    "report_id": report.report_id,
                    "updated": False,
                    "error": "Failed to save report to knowledge graph"
                }
                
        except Exception as e:
            self.logger.error(f"Error refining report {report.report_id}: {e}")
            return {
                "status": "error",
                "report_id": report.report_id,
                "updated": False,
                "error": str(e)
            }
    
    async def generate_report(self, memories: List[Dict[str, Any]]) -> DreamReport:
        """
        Generate a dream report from a set of memories.
        
        Analyzes memories to extract insights, questions, hypotheses, and counterfactuals.
        Assigns confidence values to each fragment and organizes them into a coherent report.
        
        Args:
            memories: List of memory objects to analyze
            
        Returns:
            A structured DreamReport object
        """
        self.logger.info(f"Generating report for {len(memories)} memories")
        
        if not memories:
            self.logger.warning("No memories provided for report generation")
            return DreamReport(
                id=str(uuid4()),
                title="Empty Report",
                creation_time=time.time(),
                fragments=[],
                related_memory_ids=[],
                metadata={"status": "empty", "reason": "No memories provided"}
            )
        
        try:
            # Extract memory content and IDs
            memory_contents = []
            memory_ids = []
            
            for memory in memories:
                try:
                    # Handle different memory formats
                    if isinstance(memory, dict):
                        if "content" in memory:
                            memory_contents.append(memory["content"])
                        elif "text" in memory:
                            memory_contents.append(memory["text"])
                        
                        # Extract memory ID
                        if "id" in memory:
                            memory_ids.append(memory["id"])
                        elif "memory_id" in memory:
                            memory_ids.append(memory["memory_id"])
                except Exception as e:
                    self.logger.error(f"Error processing memory: {e}")
            
            # Generate a title for the report
            amalgamated_content = "\n".join(memory_contents)
            title_prompt = f"Generate a concise, descriptive title (5-8 words) for a dream report based on these memories:\n{amalgamated_content[:1000]}..."
            
            title_response = await self.llm_service.generate_text(title_prompt)
            report_title = title_response.strip()
            
            # Truncate if title is too long
            if len(report_title) > 100:
                report_title = report_title[:97] + "..."
            
            # Generate insights, questions, hypotheses, and counterfactuals
            reflection_prompt = REFLECTION_PROMPT.format(
                memories="\n\n".join([f"Memory {i+1}: {content}" for i, content in enumerate(memory_contents)])
            )
            
            reflection_response = await self.llm_service.generate_text(reflection_prompt, format="json")
            
            # Parse the response JSON
            try:
                reflection_data = json.loads(reflection_response)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing reflection response: {e}\nResponse: {reflection_response}")
                # Attempt to fix common JSON issues (missing quotes, trailing commas)
                cleaned_response = reflection_response.replace("'\n", "\n")
                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
                
                try:
                    reflection_data = json.loads(cleaned_response)
                except:
                    # Use a default structure if parsing fails
                    reflection_data = {
                        "insights": ["Could not parse reflection response"],
                        "questions": ["What information is missing from these memories?"],
                        "hypotheses": ["The system may need review"],
                        "counterfactuals": ["What if the data was presented differently?"]
                    }
            
            # Create fragments from the reflection data
            fragments = []
            created_time = time.time()
            
            # Process insights
            for insight in reflection_data.get("insights", []):
                fragments.append(DreamFragment(
                    id=str(uuid4()),
                    type="insight",
                    content=insight,
                    creation_time=created_time,
                    confidence=0.8,  # Default high confidence for insights
                    metadata={"source": "initial_reflection"}
                ))
            
            # Process questions
            for question in reflection_data.get("questions", []):
                fragments.append(DreamFragment(
                    id=str(uuid4()),
                    type="question",
                    content=question,
                    creation_time=created_time,
                    confidence=0.6,  # Medium confidence for questions
                    metadata={"source": "initial_reflection"}
                ))
            
            # Process hypotheses
            for hypothesis in reflection_data.get("hypotheses", []):
                fragments.append(DreamFragment(
                    id=str(uuid4()),
                    type="hypothesis",
                    content=hypothesis,
                    creation_time=created_time,
                    confidence=0.5,  # Lower confidence for hypotheses
                    metadata={"source": "initial_reflection"}
                ))
            
            # Process counterfactuals
            for counterfactual in reflection_data.get("counterfactuals", []):
                fragments.append(DreamFragment(
                    id=str(uuid4()),
                    type="counterfactual",
                    content=counterfactual,
                    creation_time=created_time,
                    confidence=0.4,  # Lower confidence for counterfactuals
                    metadata={"source": "initial_reflection"}
                ))
            
            # Now generate action items based on the fragments
            action_items_prompt = ACTION_ITEMS_PROMPT.format(
                fragments="\n\n".join([f"{f.type.capitalize()}: {f.content}" for f in fragments])
            )
            
            action_response = await self.llm_service.generate_text(action_items_prompt, format="json")
            
            try:
                action_data = json.loads(action_response)
                for action in action_data.get("action_items", []):
                    fragments.append(DreamFragment(
                        id=str(uuid4()),
                        type="action",
                        content=action,
                        creation_time=created_time,
                        confidence=0.7,  # High confidence for actions
                        metadata={"source": "action_generation"}
                    ))
            except json.JSONDecodeError:
                self.logger.error(f"Error parsing action items response: {action_response}")
            
            # Generate a self-assessment
            assessment_prompt = SELF_ASSESSMENT_PROMPT.format(
                fragments="\n\n".join([f"{f.type.capitalize()}: {f.content}" for f in fragments])
            )
            
            assessment_response = await self.llm_service.generate_text(assessment_prompt, format="json")
            
            try:
                assessment_data = json.loads(assessment_response)
                for assessment in assessment_data.get("assessments", []):
                    fragments.append(DreamFragment(
                        id=str(uuid4()),
                        type="assessment",
                        content=assessment["content"],
                        creation_time=created_time,
                        confidence=float(assessment.get("confidence", 0.6)),
                        metadata={"source": "self_assessment", "category": assessment.get("category", "general")}
                    ))
            except json.JSONDecodeError:
                self.logger.error(f"Error parsing assessment response: {assessment_response}")
            
            # Create the dream report
            report = DreamReport(
                id=str(uuid4()),
                title=report_title,
                creation_time=created_time,
                fragments=fragments,
                related_memory_ids=memory_ids,
                metadata={
                    "memory_count": len(memories),
                    "fragment_count": len(fragments),
                    "confidence_avg": sum(f.confidence for f in fragments) / len(fragments) if fragments else 0
                }
            )
            
            # Store the report in the knowledge graph
            await self._store_report_in_knowledge_graph(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            # Create a minimal report in case of error
            return DreamReport(
                id=str(uuid4()),
                title="Error Report",
                creation_time=time.time(),
                fragments=[DreamFragment(
                    id=str(uuid4()),
                    type="error",
                    content=f"Error generating report: {str(e)}",
                    creation_time=time.time(),
                    confidence=0.1,
                    metadata={"error": str(e)}
                )],
                related_memory_ids=memory_ids if 'memory_ids' in locals() else [],
                metadata={"status": "error", "error": str(e)}
            )
    
    async def _get_fragments_by_ids(self, fragment_ids: List[str]) -> List[DreamFragment]:
        """
        Retrieve fragment objects by their IDs from the knowledge graph.
        
        Args:
            fragment_ids: List of fragment IDs to retrieve
            
        Returns:
            List of retrieved DreamFragment objects
        """
        fragments = []
        
        for fragment_id in fragment_ids:
            try:
                node = await self.knowledge_graph.get_node(fragment_id)
                if node and "attributes" in node:
                    fragment = DreamFragment.from_dict(node["attributes"])
                    fragments.append(fragment)
            except Exception as e:
                self.logger.error(f"Error retrieving fragment {fragment_id}: {e}")
        
        return fragments
    
    def _find_fragment_id_by_content(self, content: str, fragments: List[DreamFragment]) -> Optional[str]:
        """
        Find a fragment ID by matching its content.
        
        Args:
            content: The content to match
            fragments: List of fragments to search through
            
        Returns:
            The fragment ID if found, None otherwise
        """
        for fragment in fragments:
            if fragment.content.strip() == content.strip():
                return fragment.id
        return None
    
    async def _update_fragment_confidence(self, fragment_id: str, new_confidence_value: float, reason: str) -> bool:
        """
        Update the confidence level of a fragment in the knowledge graph.
        
        Uses a weighted approach to balance existing confidence with new evidence.
        
        Args:
            fragment_id: ID of the fragment to update
            new_confidence_value: New confidence value from evidence
            reason: Reason for the confidence adjustment
            
        Returns:
            Success status
        """
        try:
            # Get the current node
            node = await self.knowledge_graph.get_node(fragment_id)
            if not node:
                return False
            
            # Get current confidence
            current_confidence = node["attributes"].get("confidence", 0.5)
            
            # Calculate new confidence using weighted approach
            # Give more weight to existing confidence (stability) while allowing for updates
            weight_existing = 0.7  # Weight for existing confidence
            weight_new = 0.3       # Weight for new evidence
            
            # Calculate weighted confidence
            weighted_confidence = (weight_existing * current_confidence) + (weight_new * new_confidence_value)
            
            # Round to 2 decimal places for clarity
            weighted_confidence = round(weighted_confidence, 2)
            
            # Update the confidence attribute
            node["attributes"]["confidence"] = weighted_confidence
            node["attributes"]["last_updated"] = time.time()
            node["attributes"]["last_update_reason"] = reason
            
            # Save the updated node
            success = await self.knowledge_graph.update_node(fragment_id, node["attributes"])
            return success
        except Exception as e:
            self.logger.error(f"Error updating fragment confidence: {e}")
            return False
    
    async def _store_report_in_knowledge_graph(self, report: DreamReport) -> None:
        """
        Store the dream report in the knowledge graph.
        
        Args:
            report: The dream report to store
        """
        try:
            # Create a node for the report
            report_node_data = {
                "id": report.id,
                "title": report.title,
                "creation_time": report.creation_time,
                "related_memory_ids": report.related_memory_ids,
                "metadata": report.metadata,
                "node_type": "dream_report"
            }
            
            # Add the report node to the knowledge graph
            success = await self.knowledge_graph.add_node(
                node_id=report.id,
                node_type="dream_report",
                attributes=report_node_data
            )
            
            if not success:
                self.logger.error(f"Failed to add report node {report.id} to knowledge graph")
                return
            
            # Add each fragment to the knowledge graph
            for fragment in report.fragments:
                # Create a node for the fragment
                fragment_node_data = {
                    "id": fragment.id,
                    "type": fragment.type,
                    "content": fragment.content,
                    "creation_time": fragment.creation_time,
                    "confidence": fragment.confidence,
                    "metadata": fragment.metadata,
                    "node_type": "dream_fragment"
                }
                
                # Add the fragment node to the knowledge graph
                success = await self.knowledge_graph.add_node(
                    node_id=fragment.id,
                    node_type="dream_fragment",
                    attributes=fragment_node_data
                )
                
                if not success:
                    self.logger.error(f"Failed to add fragment node {fragment.id} to knowledge graph")
                    continue
                
                # Create a relationship between the report and the fragment
                success = await self.knowledge_graph.add_edge(
                    source_id=report.id,
                    target_id=fragment.id,
                    edge_type="contains_fragment",
                    attributes={
                        "created_at": time.time(),
                        "fragment_type": fragment.type
                    }
                )
                
                if not success:
                    self.logger.error(f"Failed to add edge between report {report.id} and fragment {fragment.id}")
            
            # Create relationships between the report and related memories
            for memory_id in report.related_memory_ids:
                # Verify the memory exists in the knowledge graph
                if await self.knowledge_graph.has_node(memory_id):
                    success = await self.knowledge_graph.add_edge(
                        source_id=report.id,
                        target_id=memory_id,
                        edge_type="based_on_memory",
                        attributes={
                            "created_at": time.time(),
                            "strength": 0.8  # Default relationship strength
                        }
                    )
                    
                    if not success:
                        self.logger.error(f"Failed to add edge between report {report.id} and memory {memory_id}")
            
            self.logger.info(f"Successfully stored report {report.id} in knowledge graph with {len(report.fragments)} fragments")
            
        except Exception as e:
            self.logger.error(f"Error storing report in knowledge graph: {e}")
    
    async def reflect_on_dream(self, dream_content: str, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reflect on a dream and its extracted insights to deepen understanding.
        
        This method analyzes a dream and its extracted insights, evaluating their
        quality, coherence, and potential integration into the knowledge graph.
        
        Args:
            dream_content: The content of the dream to reflect on
            insights: List of insights extracted from the dream
            
        Returns:
            Dictionary containing reflection results and meta-analysis
        """
        self.logger.info("Reflecting on dream content and insights")
        
        try:
            if not dream_content or not insights:
                self.logger.warning("Cannot reflect on empty dream or insights")
                return {
                    "status": "error",
                    "message": "Insufficient content for reflection",
                    "reflection": ""
                }
            
            # Prepare reflection context
            insight_summaries = []
            avg_significance = 0.0
            valid_insights_count = 0
            
            for idx, insight in enumerate(insights):
                # Handle different insight formats
                if isinstance(insight, str):
                    # If insight is a string, use it directly
                    insight_text = insight
                    significance = 0.5  # Default significance
                    insight_summaries.append(f"Insight {idx+1}: {insight_text} (Significance: {significance:.2f})")
                    avg_significance += significance
                    valid_insights_count += 1
                elif isinstance(insight, dict):
                    # If insight is a dictionary, check different possible structures
                    if 'attributes' in insight:
                        # Format with nested attributes
                        attributes = insight.get("attributes", {})
                        content = attributes.get('content', '')
                        significance = attributes.get('significance', 0.5)
                    else:
                        # Format with direct keys
                        content = insight.get('content', '')
                        significance = insight.get('significance', 0.5)
                    
                    if content:  # Only count insights with actual content
                        insight_summaries.append(f"Insight {idx+1}: {content} (Significance: {significance:.2f})")
                        avg_significance += significance
                        valid_insights_count += 1
            
            if valid_insights_count == 0:
                valid_insights_count = 1  # Avoid division by zero
            
            insight_text = "\n".join(insight_summaries)
            
            # Structure the reflection prompt
            reflection_prompt = f"""Dream Content:
{dream_content}

Extracted Insights:
{insight_text}

Reflection Task: Analyze the dream content and the extracted insights. 

1. Evaluate the coherence and meaning of the dream
2. Assess the quality and relevance of the extracted insights
3. Identify any missed opportunities or alternative interpretations
4. Suggest potential connections to existing knowledge
5. Provide a meta-cognitive assessment of this dream processing
"""

            # Use the language model to generate reflection
            if self.llm_service:
                reflection_response = await self.llm_service.generate(
                    prompt=reflection_prompt,
                    max_tokens=1024,
                    temperature=0.7
                )
                reflection_text = reflection_response.get("text", "")
            else:
                self.logger.warning("No LLM service available for dream reflection")
                reflection_text = "Reflection unavailable: LLM service not configured"
            
            # Store the reflection result
            result = {
                "status": "success",
                "timestamp": time.time(),
                "dream_length": len(dream_content),
                "num_insights": valid_insights_count,
                "reflection": reflection_text,
                "metadata": {
                    "avg_insight_significance": avg_significance / valid_insights_count,
                    "reflection_length": len(reflection_text)
                }
            }
            
            # Update reflection stats
            self.review_stats["total_refinements"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during dream reflection: {e}")
            return {
                "status": "error",
                "message": f"Reflection error: {str(e)}",
                "reflection": ""
            }