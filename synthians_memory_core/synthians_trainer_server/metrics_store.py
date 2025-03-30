# synthians_trainer_server/metrics_store.py

import time
import logging
import json
import datetime
import threading
import os
import math
from typing import Dict, List, Any, Optional, Union, Deque
from collections import deque, defaultdict

# Replace NumPy with pure Python implementations
def calculate_norm(vector):
    """Calculate the Euclidean norm of a vector."""
    return math.sqrt(sum(x*x for x in vector))

def calculate_mean(values):
    """Calculate the mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)

logger = logging.getLogger(__name__)

class MetricsStore:
    """Captures and stores cognitive flow metrics for introspection and diagnostics.
    
    This lightweight metrics collection system records data about memory operations,
    surprise signals, and emotional feedback to enable real-time diagnostics of
    Lucidia's cognitive processes without requiring complex UI infrastructure.
    
    The store maintains an in-memory buffer of recent metrics while offering
    optional persistence to log files for post-session analysis.
    """
    
    def __init__(self, max_buffer_size: int = 1000, 
                intent_graph_enabled: bool = True,
                log_dir: Optional[str] = None):
        """Initialize the metrics store.
        
        Args:
            max_buffer_size: Maximum number of events to keep in memory
            intent_graph_enabled: Whether to generate IntentGraph logs
            log_dir: Directory to save logs (None = no file logging)
        """
        self.max_buffer_size = max_buffer_size
        self.intent_graph_enabled = intent_graph_enabled
        self.log_dir = log_dir
        
        # Create log directory if needed
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Created metrics log directory: {self.log_dir}")
        
        # In-memory metric buffers (thread-safe)
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._memory_updates = deque(maxlen=max_buffer_size)  # Update events
        self._retrievals = deque(maxlen=max_buffer_size)  # Retrieval events
        self._quickrecal_boosts = deque(maxlen=max_buffer_size)  # QuickRecal boost events
        self._emotion_metrics = deque(maxlen=max_buffer_size)  # Emotional response events
        
        # Track current intent/interaction session
        self._current_intent_id = None
        self._intent_graph_buffer = {}
        
        # Emotional state tracking
        self._emotion_counts = defaultdict(int)
        self._user_emotion_matches = [0, 0]  # [matches, total]
        
        logger.info(f"MetricsStore initialized with buffer size {max_buffer_size}")
    
    def begin_intent(self, intent_id: Optional[str] = None) -> str:
        """Start a new intent/interaction tracking session.
        
        Returns:
            str: The intent_id (generated if not provided)
        """
        with self._lock:
            # Generate ID if not provided
            if not intent_id:
                intent_id = f"intent_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(self):x}"
            
            self._current_intent_id = intent_id
            
            # Initialize intent graph for this session
            if self.intent_graph_enabled:
                self._intent_graph_buffer[intent_id] = {
                    "trace_id": intent_id,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "memory_trace": {"retrieved": []},
                    "neural_memory_trace": {},
                    "emotional_modulation": {},
                    "reasoning_steps": [],
                    "final_output": {}
                }
            
            logger.debug(f"Started new intent tracking: {intent_id}")
            return intent_id
    
    def log_memory_update(self, input_embedding: List[float], loss: float, grad_norm: float, 
                        emotion: Optional[str] = None, intent_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics from a memory update operation.
        
        Args:
            input_embedding: The embedding that was sent to update memory
            loss: The loss value from the memory update
            grad_norm: The gradient norm from the memory update
            emotion: Optional emotion tag associated with this update
            intent_id: Optional intent ID (uses current if not provided)
            metadata: Additional metadata to store with the update
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        # Calculate embedding norm for reference
        embedding_norm = calculate_norm(input_embedding)
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "embedding_norm": embedding_norm,
            "embedding_dim": len(input_embedding),
            "emotion": emotion,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._memory_updates.append(event)
            
            # Update emotion counts if provided
            if emotion:
                self._emotion_counts[emotion] += 1
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                self._intent_graph_buffer[intent_id]["neural_memory_trace"] = {
                    **self._intent_graph_buffer[intent_id].get("neural_memory_trace", {}),
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "timestamp": event_time.isoformat()
                }
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Updated Neural Memory with new embedding (loss={loss:.4f}, grad_norm={grad_norm:.4f})"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("memory_updates", event)
        logger.debug(f"Logged memory update: loss={loss:.4f}, grad_norm={grad_norm:.4f}")
    
    def log_quickrecal_boost(self, memory_id: str, base_score: float, boost_amount: float,
                           emotion: Optional[str] = None, surprise_source: str = "neural_memory",
                           intent_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a QuickRecal score boost event.
        
        Args:
            memory_id: ID of the memory whose QuickRecal score was boosted
            base_score: Original QuickRecal score before boost
            boost_amount: Amount the score was boosted by
            emotion: Emotion associated with this memory/boost
            surprise_source: Source of the surprise signal (neural_memory, direct, etc.)
            intent_id: Optional intent ID (uses current if not provided)
            metadata: Additional metadata to store with the boost event
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "memory_id": memory_id,
            "base_score": float(base_score),
            "boost_amount": float(boost_amount),
            "final_score": float(base_score + boost_amount),
            "emotion": emotion,
            "surprise_source": surprise_source,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._quickrecal_boosts.append(event)
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                # Add to memory trace
                memory_trace = self._intent_graph_buffer[intent_id]["memory_trace"]
                memory_trace["boost_applied"] = boost_amount
                
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Boosted memory {memory_id} QuickRecal by {boost_amount:.4f} due to surprise"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("quickrecal_boosts", event)
        logger.debug(f"Logged QuickRecal boost: memory={memory_id}, amount={boost_amount:.4f}")
    
    def log_retrieval(self, query_embedding: List[float], retrieved_memories: List[Dict[str, Any]],
                     user_emotion: Optional[str] = None, intent_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a memory retrieval operation.
        
        Args:
            query_embedding: Embedding used for retrieval
            retrieved_memories: List of retrieved memories with their metadata
            user_emotion: Current user emotion if known
            intent_id: Optional intent ID (uses current if not provided) 
            metadata: Additional metadata to store with the retrieval
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        # Extract memory emotions if available
        memory_emotions = []
        for mem in retrieved_memories:
            if "dominant_emotion" in mem and mem["dominant_emotion"]:
                memory_emotions.append(mem["dominant_emotion"])
        
        # Calculate emotion match rate if user emotion is known
        emotion_match = False
        if user_emotion and memory_emotions:
            emotion_match = user_emotion in memory_emotions
            with self._lock:
                self._user_emotion_matches[0] += 1 if emotion_match else 0
                self._user_emotion_matches[1] += 1
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "embedding_dim": len(query_embedding),
            "num_results": len(retrieved_memories),
            "memory_ids": [m.get("memory_id", "unknown") for m in retrieved_memories],
            "memory_emotions": memory_emotions,
            "user_emotion": user_emotion,
            "emotion_match": emotion_match,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._retrievals.append(event)
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                memory_trace = self._intent_graph_buffer[intent_id]["memory_trace"]
                # Add retrieved memories
                memory_trace["retrieved"] = [
                    {
                        "memory_id": mem.get("memory_id", "unknown"),
                        "quickrecal_score": mem.get("quickrecal_score", 0.0),
                        "dominant_emotion": mem.get("dominant_emotion", None),
                        "emotion_confidence": mem.get("emotion_confidence", 0.0)
                    } for mem in retrieved_memories
                ]
                
                # Add emotion info if available
                if user_emotion or memory_emotions:
                    emo_mod = self._intent_graph_buffer[intent_id]["emotional_modulation"]
                    emo_mod["user_emotion"] = user_emotion
                    if memory_emotions:
                        # Find most frequent emotion
                        from collections import Counter
                        counts = Counter(memory_emotions)
                        dominant = counts.most_common(1)[0][0] if counts else None
                        emo_mod["retrieved_emotion_dominance"] = dominant
                        emo_mod["conflict_flag"] = user_emotion != dominant if user_emotion and dominant else False
                
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Retrieved {len(retrieved_memories)} memories based on query"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("retrievals", event)
        logger.debug(f"Logged retrieval: {len(retrieved_memories)} memories retrieved")
    
    def get_intent_statistics(self, intent_id: Optional[str] = None, emotion_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for a specific intent session.
        
        Args:
            intent_id: Optional intent ID (uses current if not provided)
            emotion_filter: Optional emotion to filter by
            
        Returns:
            Dict containing summary statistics
        """
        intent_id = intent_id or self._current_intent_id
        if not intent_id:
            return {}
        
        with self._lock:
            # Gather all events for this intent
            memory_updates = [e for e in self._memory_updates if e.get("intent_id") == intent_id]
            retrievals = [e for e in self._retrievals if e.get("intent_id") == intent_id]
            quickrecal_boosts = [e for e in self._quickrecal_boosts if e.get("intent_id") == intent_id]
            
            # Apply emotion filter if provided
            if emotion_filter:
                memory_updates = [e for e in memory_updates if e.get("emotion") == emotion_filter]
                retrievals = [e for e in retrievals if e.get("user_emotion") == emotion_filter]
                quickrecal_boosts = [e for e in quickrecal_boosts if e.get("emotion") == emotion_filter]
            
            # Calculate average metrics
            avg_loss = calculate_mean([e["loss"] for e in memory_updates]) if memory_updates else 0.0
            avg_grad_norm = calculate_mean([e["grad_norm"] for e in memory_updates]) if memory_updates else 0.0
            avg_boost = calculate_mean([e["boost_amount"] for e in quickrecal_boosts]) if quickrecal_boosts else 0.0
            
            # Count unique memories
            retrieved_memories = set()
            for r in retrievals:
                for mem in r.get("memory_ids", []):
                    retrieved_memories.add(mem)
            
            # Count emotions if any
            emotions = {}
            for e in memory_updates:
                if e.get("emotion"):
                    emotions[e["emotion"]] = emotions.get(e["emotion"], 0) + 1
            
            # Calculate emotion entropy if emotions present
            emotion_entropy = 0.0
            if emotions:
                total = sum(emotions.values())
                if total > 0:
                    probs = [count / total for count in emotions.values()]
                    entropy = -sum(p * math.log(p) for p in probs if p > 0)
                    emotion_entropy = float(entropy)
            
            return {
                "intent_id": intent_id,
                "event_counts": {
                    "memory_updates": len(memory_updates),
                    "retrievals": len(retrievals),
                    "quickrecal_boosts": len(quickrecal_boosts),
                },
                "metrics": {
                    "avg_loss": float(avg_loss),
                    "avg_grad_norm": float(avg_grad_norm),
                    "avg_quickrecal_boost": float(avg_boost),
                    "emotion_entropy": emotion_entropy,
                },
                "memory_stats": {
                    "unique_memories_retrieved": len(retrieved_memories),
                },
                "emotions": emotions,
            }
    
    def _maybe_write_event_log(self, event_type: str, event: Dict[str, Any]) -> None:
        """Write event to log file if logging is enabled."""
        if not self.log_dir:
            return
        
        try:
            log_file = os.path.join(self.log_dir, f"{event_type}.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write event log: {e}")
    
    def finalize_intent(self, intent_id: Optional[str] = None, 
                       response_text: Optional[str] = None,
                       confidence: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Finalize the current intent/interaction and return its IntentGraph.
        
        Args:
            intent_id: Optional intent ID (uses current if not provided)
            response_text: Final response text if available
            confidence: Confidence score for the response
            
        Returns:
            Optional[Dict[str, Any]]: The completed IntentGraph or None if not enabled
        """
        intent_id = intent_id or self._current_intent_id
        if not intent_id or not self.intent_graph_enabled:
            return None
        
        with self._lock:
            if intent_id not in self._intent_graph_buffer:
                logger.warning(f"Cannot finalize unknown intent: {intent_id}")
                return None
            
            # Complete the intent graph
            intent_graph = self._intent_graph_buffer[intent_id]
            
            # Add final output
            if response_text:
                intent_graph["final_output"] = {
                    "response_text": response_text,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            
            # Write to file if logging enabled
            if self.log_dir:
                log_file = os.path.join(self.log_dir, "intent_graphs", f"{intent_id}.json")
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                try:
                    with open(log_file, "w") as f:
                        json.dump(intent_graph, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to write intent graph: {e}")
            
            # Remove from buffer to free memory
            graph_copy = intent_graph.copy()
            del self._intent_graph_buffer[intent_id]
            
            logger.info(f"Finalized intent {intent_id} with {len(intent_graph['reasoning_steps'])} reasoning steps")
            return graph_copy
    
    def get_diagnostic_metrics(self, window: str = "last_100", 
                             emotion_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get diagnostic metrics for the emotional feedback loop.
        
        Args:
            window: Time/count window to analyze ("last_100", "last_hour", etc.)
            emotion_filter: Optional filter to specific emotion
            
        Returns:
            Dict[str, Any]: Diagnostic metrics for the emotional feedback loop
        """
        with self._lock:
            # Determine slice of data to analyze based on window
            memory_updates = list(self._memory_updates)
            quickrecal_boosts = list(self._quickrecal_boosts)
            retrievals = list(self._retrievals)
            
            # Filter by time window if needed
            if window.startswith("last_") and window[5:].isdigit():
                # "last_N" format - take last N items
                count = int(window[5:])
                memory_updates = memory_updates[-count:] if len(memory_updates) > count else memory_updates
                quickrecal_boosts = quickrecal_boosts[-count:] if len(quickrecal_boosts) > count else quickrecal_boosts
                retrievals = retrievals[-count:] if len(retrievals) > count else retrievals
            elif window == "last_hour":
                # Last hour - filter by timestamp
                cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
                cutoff_str = cutoff.isoformat()
                memory_updates = [e for e in memory_updates if e["timestamp"] >= cutoff_str]
                quickrecal_boosts = [e for e in quickrecal_boosts if e["timestamp"] >= cutoff_str]
                retrievals = [e for e in retrievals if e["timestamp"] >= cutoff_str]
            
            # Apply emotion filter if specified
            if emotion_filter and emotion_filter != "all":
                memory_updates = [e for e in memory_updates if e.get("emotion") == emotion_filter]
                quickrecal_boosts = [e for e in quickrecal_boosts if e.get("emotion") == emotion_filter]
            
            # Calculate average metrics
            avg_loss = calculate_mean([e["loss"] for e in memory_updates]) if memory_updates else 0.0
            avg_grad_norm = calculate_mean([e["grad_norm"] for e in memory_updates]) if memory_updates else 0.0
            avg_boost = calculate_mean([e["boost_amount"] for e in quickrecal_boosts]) if quickrecal_boosts else 0.0
            
            # Find dominant emotions boosted
            emotion_boost_counts = defaultdict(float)
            for e in quickrecal_boosts:
                if e.get("emotion"):
                    emotion_boost_counts[e["emotion"]] += e["boost_amount"]
            
            # Sort by boost amount and take top 5
            dominant_emotions = sorted(emotion_boost_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            dominant_emotions = [e[0] for e in dominant_emotions if e[1] > 0]
            
            # Calculate emotion entropy (diversity measure)
            emotion_counts = {k: v for k, v in self._emotion_counts.items() if v > 0}
            total_emotions = sum(emotion_counts.values())
            if total_emotions > 0:
                probs = [count/total_emotions for count in emotion_counts.values()]
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
            else:
                entropy = 0.0
            
            # Calculate user emotion match rate
            match_rate = self._user_emotion_matches[0] / self._user_emotion_matches[1] \
                if self._user_emotion_matches[1] > 0 else 0.0
            
            # Find cluster hotspots (memory IDs with most updates)
            memory_update_counts = defaultdict(int)
            for e in quickrecal_boosts:
                memory_id = e["memory_id"]
                if memory_id:
                    memory_update_counts[memory_id] += 1
            
            # Get top clusters by update count
            cluster_hotspots = sorted(memory_update_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            cluster_hotspots = [{"cluster_id": cid, "updates": count} for cid, count in cluster_hotspots if count > 0]
            
            # Generate alerts based on metrics
            alerts = []
            recommendations = []
            
            # Alerts
            if entropy < 2.0 and total_emotions > 10:
                alerts.append("⚠️ Low emotional diversity detected (entropy < 2.0)")
                recommendations.append("Introduce more varied emotional inputs")
            else:
                alerts.append("✓ Emotional diversity stable.")
                
            if avg_loss > 0.2:
                alerts.append("⚠️ High average loss detected (> 0.2)")
                recommendations.append("Check for instability in memory patterns")
            else:
                alerts.append("✓ Surprise signals healthy.")
                
            if avg_grad_norm > 1.0:
                alerts.append("⚠️ High average gradient norm (> 1.0)")
                recommendations.append("Consider reducing learning rate or checking for oscillations")
            elif avg_grad_norm > 0.5:
                alerts.append("ℹ️ Grad norm average slightly elevated.")
                recommendations.append("Monitor grad norm trend.")
            
            if match_rate < 0.5 and self._user_emotion_matches[1] > 10:
                alerts.append("⚠️ Low user emotion match rate (< 50%)")
                recommendations.append("Review emotional alignment in retrieval process")
            
            # Add generic recommendation if list is empty
            if not recommendations:
                recommendations.append("Continue monitoring with current settings")
            
            # Calculate emotion bias index (0 = balanced, 1 = highly biased)
            if len(emotion_counts) > 1 and total_emotions > 0:
                max_count = max(emotion_counts.values())
                emotion_bias = (max_count / total_emotions) * (1 - 1/len(emotion_counts))
            else:
                emotion_bias = 0.0
            
            return {
                "diagnostic_window": window,
                "avg_loss": float(avg_loss),
                "avg_grad_norm": float(avg_grad_norm),
                "avg_quickrecal_boost": float(avg_boost),
                "dominant_emotions_boosted": dominant_emotions,
                "emotional_entropy": float(entropy),
                "emotion_bias_index": float(emotion_bias),
                "user_emotion_match_rate": float(match_rate),
                "cluster_update_hotspots": cluster_hotspots,
                "alerts": alerts,
                "recommendations": recommendations,
                "data_points": {
                    "memory_updates": len(memory_updates),
                    "quickrecal_boosts": len(quickrecal_boosts),
                    "retrievals": len(retrievals)
                }
            }
    
    def format_diagnostics_as_table(self, diagnostics: Dict[str, Any]) -> str:
        """Format diagnostics as an ASCII table for CLI output.
        
        Args:
            diagnostics: Diagnostics data from get_diagnostic_metrics()
            
        Returns:
            str: Formatted ASCII table
        """
        width = 80
        
        # Helper to create a section line
        def section(title):
            return f"\n{title.center(width, '=')}\n"
        
        # Ensure diagnostics dict has all expected keys with defaults
        defaults = {
            'diagnostic_window': 'Unknown',
            'avg_loss': 0.0,
            'avg_grad_norm': 0.0,
            'avg_quickrecal_boost': 0.0,
            'emotional_entropy': 0.0,
            'emotion_bias_index': 0.0,
            'user_emotion_match_rate': 0.0,
            'dominant_emotions_boosted': [],
            'cluster_update_hotspots': [],
            'alerts': [],
            'recommendations': [], 
            'data_points': {
                'memory_updates': 0,
                'quickrecal_boosts': 0,
                'retrievals': 0
            }
        }
        
        # Fill in any missing keys with defaults
        for key, default_value in defaults.items():
            if key not in diagnostics:
                diagnostics[key] = default_value
                # Only log warning for non-standard missing keys
                if key != 'data_points':  # data_points is commonly missing and handled with defaults
                    logger.warning(f"Missing key '{key}' in diagnostics, using default value")
                
        # Ensure data_points has all expected keys
        if 'data_points' not in diagnostics:
            diagnostics['data_points'] = {}
            
        for key, default_value in defaults['data_points'].items():
            if key not in diagnostics['data_points']:
                diagnostics['data_points'][key] = default_value
                # Only log warning for non-standard missing keys at debug level
                logger.debug(f"Missing key '{key}' in data_points, using default value")
        
        # Header
        output = []
        output.append("=" * width)
        output.append(f"LUCIDIA COGNITIVE DIAGNOSTICS: {diagnostics['diagnostic_window']}".center(width))
        output.append(f"[{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}]".center(width))
        output.append("=" * width)
        
        # Core metrics
        output.append(section("CORE METRICS"))
        metrics = [
            ("Average Loss", f"{diagnostics['avg_loss']:.4f}"),
            ("Average Grad Norm", f"{diagnostics['avg_grad_norm']:.4f}"),
            ("Average QuickRecal Boost", f"{diagnostics['avg_quickrecal_boost']:.4f}"),
            ("Emotional Entropy", f"{diagnostics['emotional_entropy']:.2f}"),
            ("Emotion Bias Index", f"{diagnostics['emotion_bias_index']:.2f}"),
            ("User Emotion Match Rate", f"{diagnostics['user_emotion_match_rate']:.2%}")
        ]
        
        # Format metrics as two columns
        for i in range(0, len(metrics), 2):
            if i+1 < len(metrics):
                col1 = f"{metrics[i][0]}: {metrics[i][1]}"
                col2 = f"{metrics[i+1][0]}: {metrics[i+1][1]}"
                output.append(f"{col1.ljust(40)} | {col2.ljust(38)}")
            else:
                output.append(f"{metrics[i][0]}: {metrics[i][1]}")
        
        # Dominant emotions
        output.append(section("EMOTION ANALYSIS"))
        if diagnostics['dominant_emotions_boosted']:
            output.append("Dominant Boosted Emotions: " + ", ".join(diagnostics['dominant_emotions_boosted']))
        else:
            output.append("Dominant Boosted Emotions: None detected")
        
        # Cluster hotspots
        output.append(section("MEMORY HOTSPOTS"))
        if diagnostics['cluster_update_hotspots']:
            for hotspot in diagnostics['cluster_update_hotspots']:
                hotspot_id = hotspot.get('cluster_id', 'Unknown')
                updates = hotspot.get('updates', 0)
                output.append(f"* {hotspot_id}: {updates} updates")
        else:
            output.append("No significant memory hotspots detected")
        
        # Alerts and recommendations
        output.append(section("ALERTS"))
        for alert in diagnostics['alerts']:
            output.append(f"* {alert}")
        
        output.append(section("RECOMMENDATIONS"))
        for rec in diagnostics['recommendations']:
            output.append(f"* {rec}")
        
        # Data summary
        data_points = diagnostics.get('data_points', {})
        output.append(section("DATA SUMMARY"))
        output.append(f"Based on {data_points.get('memory_updates', 0)} updates, {data_points.get('quickrecal_boosts', 0)} boosts, and {data_points.get('retrievals', 0)} retrievals")
        output.append("=" * width)  
        
        return "\n".join(output)
    
# --- Global Instance ---
metrics_store = None

def get_metrics_store() -> MetricsStore:
    """Get or initialize the global MetricsStore instance."""
    global metrics_store
    if metrics_store is None:
        # Create log directory in the current directory
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        metrics_store = MetricsStore(log_dir=log_dir)
        logger.info("Global MetricsStore initialized")
    return metrics_store
