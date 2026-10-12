from ie.schema import ExtractedField
from ie.correction.rules import DateNormalizationRule, ContractNumberRule, CompanyAliasRule

class CorrectionLayer:
    def __init__(self):
        self.rules = {
            "date": DateNormalizationRule(),
            "contract_number": ContractNumberRule(),
            "recipient_company": CompanyAliasRule()
        }

    def correct_field(self, field_name: str, field: ExtractedField, context: dict = None) -> ExtractedField:
        # Clone field to avoid mutating original
        new_field = ExtractedField(
            value=field.value,
            source_text=field.source_text,
            method=field.method,
            status=field.status,
            confidence=field.confidence,
            original_value=field.value
        )
        
        # Don't correct if it's already failed or empty
        if field.status == "failed" or not field.value:
            return new_field
            
        rule = self.rules.get(field_name)
        if not rule:
            return new_field
            
        applied, corrected_value, reason = rule.apply(field.value, context)
        
        if applied and corrected_value != field.value:
            new_field.value = corrected_value
            new_field.corrected_value = corrected_value
            new_field.correction_applied = True
            new_field.correction_method = rule.__class__.__name__
            new_field.correction_reason = reason
            
        return new_field

    def correct_document(self, extracted_fields: dict[str, ExtractedField], context: dict = None) -> dict[str, ExtractedField]:
        corrected_fields = {}
        for field_name, field in extracted_fields.items():
            corrected_fields[field_name] = self.correct_field(field_name, field, context)
        return corrected_fields
