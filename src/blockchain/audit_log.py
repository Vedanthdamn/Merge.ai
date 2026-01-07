"""
Blockchain-Based Audit Logging for Federated Learning

Blockchain provides an immutable, transparent audit trail for distributed
machine learning systems.

Purpose:
1. Record all training rounds and participating hospitals
2. Log model version hashes for reproducibility
3. Ensure transparency and accountability
4. Enable auditing without compromising data privacy

Key Properties:
- Immutability: Past records cannot be altered
- Transparency: All stakeholders can verify the training history
- Decentralization: No single entity controls the audit log

Important: Blockchain is used ONLY for audit logging, NOT for:
- Training (computationally expensive)
- Inference (unnecessary overhead)
- Data storage (data remains at hospitals)

Use Case Example:
"Which hospitals contributed to model version 3.2?"
→ Blockchain provides verifiable answer

Note: This is a simplified implementation for demonstration.
Production systems would use established blockchain frameworks.
"""

import hashlib
import json
import time
from typing import List, Dict, Optional
from datetime import datetime


class Block:
    """
    A single block in the blockchain.
    
    Each block contains:
    - Index: Position in chain
    - Timestamp: When block was created
    - Data: Information about the training round
    - Previous hash: Link to previous block (ensures immutability)
    - Hash: Unique identifier of this block
    """
    
    def __init__(self, index: int, timestamp: float, data: Dict,
                 previous_hash: str = "0"):
        """
        Initialize a block.
        
        Args:
            index (int): Block number in chain
            timestamp (float): Unix timestamp
            data (dict): Block data (training round info)
            previous_hash (str): Hash of previous block
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate cryptographic hash of block.
        
        Hash includes all block data, making tampering detectable.
        If any data changes, hash changes, breaking the chain.
        
        Returns:
            SHA-256 hash of block contents
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash
        }


class FederatedLearningBlockchain:
    """
    Blockchain for auditing federated learning training.
    
    Records:
    - Training rounds
    - Participating hospitals
    - Model version hashes
    - Aggregation results
    - Performance metrics
    
    This creates a transparent, immutable record of the entire
    training process without storing sensitive patient data.
    """
    
    def __init__(self):
        """Initialize blockchain with genesis block."""
        self.chain: List[Block] = []
        self.create_genesis_block()
        print("[Blockchain] Initialized with genesis block")
    
    def create_genesis_block(self):
        """
        Create the first block in the chain.
        
        The genesis block has no previous block and serves as the
        foundation of the blockchain.
        """
        genesis_data = {
            "event": "blockchain_initialized",
            "description": "Federated Learning Audit Log Started",
            "system": "Healthcare Privacy-Preserving ML"
        }
        
        genesis_block = Block(0, time.time(), genesis_data, "0")
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]
    
    def add_block(self, data: Dict) -> Block:
        """
        Add a new block to the chain.
        
        Args:
            data (dict): Data to store in block
            
        Returns:
            The newly created block
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=latest_block.hash
        )
        self.chain.append(new_block)
        
        print(f"[Blockchain] Block {new_block.index} added: {data.get('event', 'unknown')}")
        return new_block
    
    def log_training_round(self, round_number: int, participating_hospitals: List[int],
                          model_hash: str, metrics: Dict, aggregation_method: str = "FedAvg"):
        """
        Log a federated learning training round.
        
        This creates a permanent, auditable record of:
        - Which hospitals participated
        - What model version was created
        - Performance metrics achieved
        - How aggregation was performed
        
        Args:
            round_number (int): FL round number
            participating_hospitals (list): List of hospital IDs
            model_hash (str): Hash of aggregated model
            metrics (dict): Performance metrics
            aggregation_method (str): Method used for aggregation
        """
        data = {
            "event": "training_round",
            "round": round_number,
            "participating_hospitals": participating_hospitals,
            "n_participants": len(participating_hospitals),
            "model_hash": model_hash,
            "aggregation_method": aggregation_method,
            "metrics": metrics,
            "privacy_preserved": True  # No raw data transmitted
        }
        
        self.add_block(data)
    
    def log_model_evaluation(self, model_hash: str, evaluation_metrics: Dict,
                            evaluated_by: Optional[int] = None):
        """
        Log model evaluation results.
        
        Args:
            model_hash (str): Hash of evaluated model
            evaluation_metrics (dict): Evaluation results
            evaluated_by (int): Hospital ID that performed evaluation (optional)
        """
        data = {
            "event": "model_evaluation",
            "model_hash": model_hash,
            "metrics": evaluation_metrics,
            "evaluated_by": evaluated_by
        }
        
        self.add_block(data)
    
    def log_model_deployment(self, model_hash: str, deployment_location: str,
                            approved_by: Optional[str] = None):
        """
        Log model deployment.
        
        Args:
            model_hash (str): Hash of deployed model
            deployment_location (str): Where model is deployed
            approved_by (str): Who approved deployment (optional)
        """
        data = {
            "event": "model_deployment",
            "model_hash": model_hash,
            "deployment_location": deployment_location,
            "approved_by": approved_by
        }
        
        self.add_block(data)
    
    def log_privacy_audit(self, audit_type: str, audit_results: Dict):
        """
        Log privacy audit results.
        
        Args:
            audit_type (str): Type of audit (e.g., "differential_privacy", "data_leakage")
            audit_results (dict): Audit findings
        """
        data = {
            "event": "privacy_audit",
            "audit_type": audit_type,
            "results": audit_results
        }
        
        self.add_block(data)
    
    def verify_chain(self) -> bool:
        """
        Verify blockchain integrity.
        
        Checks:
        1. Each block's hash is correct
        2. Each block properly links to previous block
        3. Chain has not been tampered with
        
        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify hash
            if current_block.hash != current_block.calculate_hash():
                print(f"[Blockchain] Invalid hash at block {i}")
                return False
            
            # Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                print(f"[Blockchain] Chain broken at block {i}")
                return False
        
        print("[Blockchain] Chain verified: All blocks are valid")
        return True
    
    def get_training_history(self) -> List[Dict]:
        """
        Get complete training history.
        
        Returns:
            List of all training round records
        """
        history = []
        for block in self.chain:
            if block.data.get("event") == "training_round":
                history.append(block.to_dict())
        return history
    
    def get_model_lineage(self, model_hash: str) -> List[Dict]:
        """
        Trace the lineage of a specific model.
        
        Finds all blocks related to a model hash.
        
        Args:
            model_hash (str): Model hash to trace
            
        Returns:
            List of relevant blocks
        """
        lineage = []
        for block in self.chain:
            if block.data.get("model_hash") == model_hash:
                lineage.append(block.to_dict())
        return lineage
    
    def print_chain(self):
        """Print the entire blockchain in human-readable format."""
        print("\n" + "="*60)
        print("BLOCKCHAIN AUDIT LOG")
        print("="*60)
        
        for block in self.chain:
            block_dict = block.to_dict()
            print(f"\n{'─'*60}")
            print(f"Block #{block_dict['index']}")
            print(f"Timestamp: {block_dict['datetime']}")
            print(f"Hash: {block_dict['hash'][:16]}...")
            print(f"Previous Hash: {block_dict['previous_hash'][:16]}...")
            print(f"Data: {json.dumps(block_dict['data'], indent=2)}")
        
        print("\n" + "="*60)
    
    def export_to_json(self, filepath: str):
        """
        Export blockchain to JSON file.
        
        Args:
            filepath (str): Output file path
        """
        chain_data = [block.to_dict() for block in self.chain]
        
        with open(filepath, 'w') as f:
            json.dump(chain_data, f, indent=2)
        
        print(f"[Blockchain] Exported to {filepath}")


def calculate_model_hash(model_parameters: List) -> str:
    """
    Calculate hash of model parameters.
    
    This creates a unique identifier for each model version,
    enabling traceability without storing entire models.
    
    Args:
        model_parameters: List of model weights/parameters
        
    Returns:
        SHA-256 hash of parameters
    """
    # Convert parameters to string representation
    param_string = json.dumps([p.tolist() if hasattr(p, 'tolist') else p 
                               for p in model_parameters])
    
    return hashlib.sha256(param_string.encode()).hexdigest()


def demonstrate_blockchain():
    """
    Demonstrate blockchain for FL audit logging.
    """
    print("="*60)
    print("Blockchain-Based Audit Logging Demonstration")
    print("="*60)
    
    # Initialize blockchain
    blockchain = FederatedLearningBlockchain()
    
    # Simulate federated learning rounds
    print("\nSimulating federated learning training...")
    
    # Round 1
    blockchain.log_training_round(
        round_number=1,
        participating_hospitals=[0, 1, 2],
        model_hash="abc123def456...",
        metrics={"accuracy": 0.75, "loss": 0.45},
        aggregation_method="FedAvg"
    )
    
    time.sleep(0.1)  # Small delay for different timestamps
    
    # Round 2
    blockchain.log_training_round(
        round_number=2,
        participating_hospitals=[0, 1, 2],
        model_hash="def789ghi012...",
        metrics={"accuracy": 0.82, "loss": 0.35},
        aggregation_method="FedAvg"
    )
    
    time.sleep(0.1)
    
    # Evaluation
    blockchain.log_model_evaluation(
        model_hash="def789ghi012...",
        evaluation_metrics={"test_accuracy": 0.81, "test_auc": 0.85},
        evaluated_by=0
    )
    
    time.sleep(0.1)
    
    # Deployment
    blockchain.log_model_deployment(
        model_hash="def789ghi012...",
        deployment_location="Hospital Network Production",
        approved_by="Medical Director"
    )
    
    time.sleep(0.1)
    
    # Privacy audit
    blockchain.log_privacy_audit(
        audit_type="differential_privacy",
        audit_results={"epsilon": 1.0, "compliant": True}
    )
    
    # Print blockchain
    blockchain.print_chain()
    
    # Verify integrity
    print("\nVerifying blockchain integrity...")
    blockchain.verify_chain()
    
    # Get training history
    print("\n" + "="*60)
    print("Training History Summary")
    print("="*60)
    history = blockchain.get_training_history()
    for record in history:
        print(f"\nRound {record['data']['round']}:")
        print(f"  Hospitals: {record['data']['participating_hospitals']}")
        print(f"  Metrics: {record['data']['metrics']}")
    
    # Export
    print("\n" + "="*60)
    blockchain.export_to_json("blockchain_audit.json")
    
    print("\n" + "="*60)
    print("Key Benefits:")
    print("="*60)
    print("✓ Immutable record of all training rounds")
    print("✓ Transparent audit trail for regulators")
    print("✓ Model lineage tracking for reproducibility")
    print("✓ No sensitive data stored (only metadata)")
    print("✓ Tamper-evident (any change breaks chain)")
    print("="*60)


def main():
    """Main demonstration function."""
    demonstrate_blockchain()


if __name__ == "__main__":
    main()
