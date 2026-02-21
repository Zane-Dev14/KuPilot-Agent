"""Incident state store for multi-turn diagnosis."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Incident(BaseModel):
    """A single incident involving pod failures."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    namespace: str = "default"
    pod_name: str = ""
    
    # Diagnosis progression
    symptoms: str = ""
    root_cause: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    
    # Proposed fix
    hypothesis: Optional[str] = None
    fix_commands: list[str] = Field(default_factory=list)
    risk_score: float = 0.5
    
    # Verification
    verified: bool = False
    status: str = "open"  # open | resolved | escalated
    
    # Chat history within incident
    messages: list[dict] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a turn to the incident conversation."""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return self.model_dump(mode="json")


class IncidentStore:
    """Simple JSON-backed incident storage."""
    
    def __init__(self, storage_dir: str = "data/incidents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, incident: Incident) -> str:
        """Save an incident and return its ID."""
        path = self.storage_dir / f"{incident.id}.json"
        with open(path, "w") as f:
            json.dump(incident.to_dict(), f, indent=2)
        return incident.id
    
    def load(self, incident_id: str) -> Optional[Incident]:
        """Load an incident by ID."""
        path = self.storage_dir / f"{incident_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return Incident(**data)
    
    def list_incidents(self, status: Optional[str] = None) -> list[Incident]:
        """List all incidents, optionally filtered by status."""
        incidents = []
        for path in self.storage_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            incident = Incident(**data)
            if status is None or incident.status == status:
                incidents.append(incident)
        return incidents
    
    def update_status(self, incident_id: str, status: str):
        """Update incident status."""
        incident = self.load(incident_id)
        if incident:
            incident.status = status
            self.save(incident)


# Global instance
_store = None


def get_incident_store() -> IncidentStore:
    """Get or create the global incident store."""
    global _store
    if _store is None:
        _store = IncidentStore()
    return _store
