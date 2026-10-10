from dataclasses import dataclass
from typing import Optional

@dataclass
class ExtractedField:
    value: str
    source_text: str
    method: str
    status: str  # e.g., "candidate", "confident", "failed"
    confidence: Optional[float] = None
    
    def to_dict(self):
        return {
            "value": self.value,
            "source_text": self.source_text,
            "method": self.method,
            "status": self.status,
            "confidence": self.confidence
        }
