#!/usr/bin/env python
"""
API-based Vector Index Repair Script

This script uses the Memory Core API to fix:
1. Vector index inconsistencies (FAISS count vs mapping count mismatch)
2. Invalid assembly data format issues (missing 'memories' field)

It works by first attempting to repair the vector index through the API,
then checking and repairing invalid assemblies.
"""

import sys
import json
import time
import logging
import argparse
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("api_repair_index")

class APIIndexRepairer:
    """Repair vector index and assemblies using the Memory Core API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health")
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                log.info("Memory Core API is healthy")
                return True
            else:
                log.error(f"Memory Core API health check failed: {resp.text}")
                return False
        except Exception as e:
            log.error(f"Error checking API health: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server stats including vector index and assembly info."""
        try:
            resp = self.session.get(f"{self.base_url}/stats")
            if resp.status_code == 200:
                data = resp.json()
                log.info("Retrieved Memory Core stats")
                return data
            else:
                log.error(f"Failed to get stats: {resp.text}")
                return {}
        except Exception as e:
            log.error(f"Error getting stats: {str(e)}")
            return {}
    
    def check_vector_index_consistency(self) -> bool:
        """Check if the vector index is consistent based on stats."""
        stats = self.get_stats()
        if not stats:
            return False
        
        memory_core = stats.get("memory_core", {})
        vector_index = memory_core.get("vector_index", {})
        
        faiss_count = vector_index.get("faiss_count", 0)
        mapping_count = vector_index.get("id_mapping_count", 0)
        is_consistent = vector_index.get("is_consistent", False)
        
        log.info(f"Vector index stats: FAISS count={faiss_count}, Mapping count={mapping_count}, Consistent={is_consistent}")
        return is_consistent
    
    def get_assembly_stats(self) -> Dict[str, Any]:
        """Get assembly-related stats."""
        stats = self.get_stats()
        if not stats:
            return {}
        
        memory_core = stats.get("memory_core", {})
        return memory_core.get("assemblies", {})
    
    def repair_vector_index(self, repair_type: str = "complete_rebuild") -> bool:
        """Attempt to repair the vector index using the API."""
        try:
            log.info(f"Attempting to repair vector index with repair_type={repair_type}")
            resp = self.session.post(f"{self.base_url}/repair_index?repair_type={repair_type}")
            
            if resp.status_code == 200:
                data = resp.json()
                log.info(f"Repair response: {json.dumps(data, indent=2)}")
                
                # Check if repair was successful
                if data.get("success", False):
                    log.info("Vector index repair was successful")
                    return True
                else:
                    log.warning(f"Vector index repair reported failure: {data.get('message', 'Unknown error')}")
                    return False
            else:
                log.error(f"Failed to repair vector index: {resp.text}")
                return False
        except Exception as e:
            log.error(f"Error repairing vector index: {str(e)}")
            return False
    
    def list_assemblies(self) -> List[Dict[str, Any]]:
        """List all assemblies via API."""
        try:
            resp = self.session.get(f"{self.base_url}/assemblies")
            if resp.status_code == 200:
                data = resp.json()
                assemblies = data.get("assemblies", [])
                log.info(f"Retrieved {len(assemblies)} assemblies")
                return assemblies
            else:
                log.error(f"Failed to list assemblies: {resp.text}")
                return []
        except Exception as e:
            log.error(f"Error listing assemblies: {str(e)}")
            return []
    
    def get_assembly(self, assembly_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific assembly."""
        try:
            resp = self.session.get(f"{self.base_url}/assemblies/{assembly_id}")
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning(f"Failed to get assembly {assembly_id}: {resp.text}")
                return None
        except Exception as e:
            log.error(f"Error getting assembly {assembly_id}: {str(e)}")
            return None
    
    def delete_assembly(self, assembly_id: str) -> bool:
        """Delete an invalid assembly."""
        try:
            resp = self.session.delete(f"{self.base_url}/assemblies/{assembly_id}")
            if resp.status_code == 200:
                log.info(f"Successfully deleted assembly {assembly_id}")
                return True
            else:
                log.warning(f"Failed to delete assembly {assembly_id}: {resp.text}")
                return False
        except Exception as e:
            log.error(f"Error deleting assembly {assembly_id}: {str(e)}")
            return False
    
    def check_and_fix_assemblies(self) -> Dict[str, int]:
        """Check all assemblies and fix or delete invalid ones."""
        stats = {"total": 0, "valid": 0, "invalid": 0, "deleted": 0, "failed": 0}
        
        # List all assemblies
        assemblies = self.list_assemblies()
        stats["total"] = len(assemblies)
        
        if not assemblies:
            log.info("No assemblies found to check")
            return stats
        
        # Check each assembly
        for assembly in assemblies:
            assembly_id = assembly.get("id")
            if not assembly_id:
                continue
            
            # Get detailed assembly data
            details = self.get_assembly(assembly_id)
            if not details:
                stats["failed"] += 1
                continue
            
            # Check for required fields
            has_memories = "memories" in details and details["memories"] is not None
            
            if has_memories:
                stats["valid"] += 1
                log.debug(f"Assembly {assembly_id} is valid")
            else:
                stats["invalid"] += 1
                log.warning(f"Assembly {assembly_id} is invalid (missing 'memories' field)")
                
                # Delete invalid assembly
                if self.delete_assembly(assembly_id):
                    stats["deleted"] += 1
        
        log.info(f"Assembly check complete: {stats['total']} total, {stats['valid']} valid, "
                f"{stats['invalid']} invalid, {stats['deleted']} deleted, {stats['failed']} failed to check")
        return stats

def repair_all(base_url: str):
    """Main function to repair both vector index and invalid assemblies."""
    repairer = APIIndexRepairer(base_url)
    
    # Step 1: Check API health
    if not repairer.check_health():
        log.error("Cannot proceed with repair - API is not healthy")
        return False
    
    # Step 2: Check initial vector index status
    log.info("Checking initial vector index status...")
    initial_status = repairer.check_vector_index_consistency()
    
    if initial_status:
        log.info("Vector index is already consistent, skipping repair")
    else:
        # Step 3: Repair vector index
        log.info("Vector index is inconsistent, attempting repair...")
        repair_success = repairer.repair_vector_index("auto")
        
        # Use more aggressive repair if needed
        if not repair_success:
            log.warning("Standard repair failed, trying complete rebuild...")
            repair_success = repairer.repair_vector_index("complete_rebuild")
        
        # Check if repair was successful
        time.sleep(2)  # Give server time to update stats
        post_repair_status = repairer.check_vector_index_consistency()
        
        if post_repair_status:
            log.info("Vector index repair was successful!")
        else:
            log.warning("Vector index is still inconsistent after repair attempts")
    
    # Step 4: Check and fix assemblies
    log.info("Checking and fixing invalid assemblies...")
    assembly_stats = repairer.check_and_fix_assemblies()
    
    if assembly_stats["invalid"] == 0 or assembly_stats["deleted"] == assembly_stats["invalid"]:
        log.info("Assembly repair complete - all invalid assemblies were processed")
    else:
        log.warning(f"Some invalid assemblies could not be fixed: "
                  f"{assembly_stats['invalid'] - assembly_stats['deleted']} remain")
    
    # Step 5: Final check
    final_vector_status = repairer.check_vector_index_consistency()
    final_assembly_stats = repairer.get_assembly_stats()
    
    log.info("Repair process complete:")
    log.info(f"  - Vector index consistent: {final_vector_status}")
    log.info(f"  - Assembly count: {final_assembly_stats.get('count', 'unknown')}")
    
    return final_vector_status

def main():
    parser = argparse.ArgumentParser(description="Repair Memory Core vector index and assemblies via API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:5010",
        help="Base URL for Memory Core API (default: http://localhost:5010)"
    )
    args = parser.parse_args()
    
    log.info(f"Starting API-based repair on {args.base_url}")
    success = repair_all(args.base_url)
    
    if success:
        log.info("Repair process completed successfully")
        return 0
    else:
        log.warning("Repair process completed with warnings or errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
