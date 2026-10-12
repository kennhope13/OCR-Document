from dataclasses import dataclass
from typing import Optional

@dataclass
class ExtractedField:
    value: str
    source_text: str
    method: str
    status: str  # e.g., "candidate", "confident", "failed"
    confidence: Optional[float] = None
    
    # Metadata for Config C (Correction Layer)
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    correction_applied: bool = False
    correction_method: Optional[str] = None
    correction_reason: Optional[str] = None
    
    def to_dict(self):
        return {
            "value": self.value,
            "source_text": self.source_text,
            "method": self.method,
            "status": self.status,
            "confidence": self.confidence,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "correction_applied": self.correction_applied,
            "correction_method": self.correction_method,
            "correction_reason": self.correction_reason,
        }
