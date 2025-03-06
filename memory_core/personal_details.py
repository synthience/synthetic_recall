# memory_core/personal_details.py

import logging
import re
from typing import Dict, Any, List, Optional, Set
import time

logger = logging.getLogger(__name__)

class PersonalDetailsMixin:
    """
    Mixin that handles personal details extraction and storage.
    Automatically detects and stores personal information with high significance.
    """
    
    def __init__(self):
        # Initialize personal details storage if not exists
        if not hasattr(self, "personal_details"):
            self.personal_details = {}
        
        # Initialize common patterns
        self._name_patterns = [
            r"(?:my name is|i am|i'm|call me) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})",
            r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,2}) (?:is my name|here)",
        ]
        
        # Known personal detail categories
        self._personal_categories = {
            "name": {"patterns": self._name_patterns, "significance": 0.9},
            "birthday": {"patterns": [r"(?:my birthday is|i was born on) (.+?)[.\n]?"], "significance": 0.85},
            "location": {"patterns": [r"(?:i live in|i'm from|i am from) (.+?)[.\n,]?"], "significance": 0.8},
            "job": {"patterns": [r"(?:i work as a|my job is|i am a) (.+?)[.\n,]?"], "significance": 0.75},
            "family": {"patterns": [r"(?:my (?:wife|husband|partner|son|daughter|child|children) (?:is|are)) (.+?)[.\n,]?"], "significance": 0.85},
        }
        
        # Initialize list of detected names
        self._detected_names: Set[str] = set()
    
    async def detect_personal_details(self, text: str) -> Dict[str, Any]:
        """
        Detect personal details in text using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict of detected personal details
        """
        found_details = {}
        
        # Check all personal categories
        for category, config in self._personal_categories.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches[0].strip()
                    found_details[category] = {
                        "value": value,
                        "confidence": 0.9,  # High confidence for direct pattern matches
                        "significance": config["significance"]
                    }
                    
                    # If we found a name, add to detected names
                    if category == "name":
                        self._detected_names.add(value.lower())
        
        return found_details
    
    async def check_for_name_references(self, text: str) -> List[str]:
        """
        Check if text refers to previously detected names.
        
        Args:
            text: The text to check
            
        Returns:
            List of detected names in the text
        """
        found_names = []
        
        # Check for each detected name in the text
        for name in self._detected_names:
            # Split the name to handle first names vs. full names
            name_parts = name.split()
            
            for part in name_parts:
                # Only check parts with length > 2 to avoid false positives
                if len(part) > 2:
                    # Look for the name with word boundaries
                    pattern = r'\b' + re.escape(part) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        found_names.append(name)
                        break
        
        return found_names
    
    async def store_personal_detail(self, category: str, value: str, significance: float = 0.8) -> bool:
        """
        Store a personal detail with high significance.
        
        Args:
            category: The type of personal detail (e.g., 'name', 'location')
            value: The value of the personal detail
            significance: Significance score (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # Store in personal details dict with clear user attribution
            self.personal_details[category] = {
                "value": value,
                "timestamp": time.time(),
                "significance": significance,
                "belongs_to": "USER"  # Explicit attribution to USER
            }
            
            # Also store as a high-significance memory with clear USER attribution
            memory_content = f"USER {category}: {value}"
            await self.store_memory(
                content=memory_content,
                significance=significance,
                metadata={
                    "type": "personal_detail",
                    "category": category,
                    "value": value,
                    "belongs_to": "USER"  # Explicit attribution
                }
            )
            
            # If it's a name, add to detected names
            if category == "name":
                self._detected_names.add(value.lower())
            
            logger.info(f"Stored USER personal detail: {category}={value}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing personal detail: {e}")
            return False
    
    async def get_personal_detail(self, category: str) -> Optional[str]:
        """
        Retrieve a personal detail by category.
        
        Args:
            category: The type of personal detail to retrieve
            
        Returns:
            The value or None if not found
        """
        detail = self.personal_details.get(category)
        if detail:
            return detail.get("value")
        return None
    
    async def process_message_for_personal_details(self, text: str) -> None:
        """
        Process an incoming message to extract and store personal details.
        
        Args:
            text: The message text to process
        """
        # Detect personal details
        details = await self.detect_personal_details(text)
        
        # Store each detected detail
        for category, detail in details.items():
            await self.store_personal_detail(
                category=category,
                value=detail["value"],
                significance=detail["significance"]
            )
        
        # Check for references to known names
        name_references = await self.check_for_name_references(text)
        
        # If names were referenced, boost their significance in memory
        if name_references:
            for name in name_references:
                # Create a memory about the name reference
                memory_content = f"USER mentioned name: {name}"
                await self.store_memory(
                    content=memory_content,
                    significance=0.75,  # Slightly lower than initial detection
                    metadata={
                        "type": "name_reference",
                        "name": name,
                        "belongs_to": "USER"  # Explicit attribution
                    }
                )
    
    async def get_personal_details_tool(self, category: str = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details.
        
        Args:
            category: Optional category of personal detail to retrieve
            
        Returns:
            Dict with personal details
        """
        # Special handling for name queries
        if category and category.lower() == "name":
            # First try to get from personal details dictionary
            value = await self.get_personal_detail("name")
            
            if value:
                logger.info(f"Retrieved USER name from personal details: {value}")
                return {
                    "found": True,
                    "category": "name",
                    "value": value,
                    "confidence": 0.95,
                    "belongs_to": "USER",  # Explicit attribution
                    "note": "This is the USER's name, not the assistant's name. The assistant's name is Lucidia."
                }
            
            # If not found in personal details, try searching memory
            try:
                # Search for name introduction memories with high significance
                if hasattr(self, "search_memory"):
                    name_memories = await self.search_memory(
                        "user name", 
                        limit=3, 
                        min_significance=0.8
                    )
                    
                    # Look for explicit name statements
                    for memory in name_memories:
                        content = memory.get("content", "")
                        
                        # Check for explicit name statements
                        name_patterns = [
                            r"USER name: ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"USER explicitly stated their name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"The USER's name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"([A-Za-z]+(?: [A-Za-z]+){0,2}) is my name"
                        ]
                        
                        for pattern in name_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                found_name = matches[0].strip()
                                logger.info(f"Found USER name in memory: {found_name}")
                                
                                # Store it in personal details for future reference
                                await self.store_personal_detail("name", found_name, 0.9)
                                
                                return {
                                    "found": True,
                                    "category": "name",
                                    "value": found_name,
                                    "confidence": 0.85,
                                    "source": "memory",
                                    "belongs_to": "USER",  # Explicit attribution
                                    "note": "This is the USER's name, not the assistant's name. The assistant's name is Lucidia."
                                }
            except Exception as e:
                logger.error(f"Error searching memory for name: {e}")
        
        # If specific category requested (and not handled by special cases above)
        if category:
            value = await self.get_personal_detail(category)
            return {
                "found": value is not None,
                "category": category,
                "value": value,
                "belongs_to": "USER",  # Explicit attribution
                "note": f"This is the USER's {category}, not the assistant's {category}."
            }
        
        # Return all personal details
        formatted_details = {}
        for cat, detail in self.personal_details.items():
            formatted_details[cat] = detail.get("value")
        
        return {
            "personal_details": formatted_details,
            "count": len(formatted_details),
            "belongs_to": "USER",  # Explicit attribution
            "note": "These are the USER's personal details, not the assistant's."
        }