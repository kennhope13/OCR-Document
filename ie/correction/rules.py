import re

class DateNormalizationRule:
    def apply(self, value: str, context: dict = None) -> tuple[bool, str, str]:
        if not value: return False, value, ""
        new_value = re.sub(r'(\d+),(\d{4})', r'\1, \2', value)
        if new_value != value:
            return True, new_value, "Normalized date spacing"
        return False, value, ""

class ContractNumberRule:
    def apply(self, value: str, context: dict = None) -> tuple[bool, str, str]:
        if not value or not context: return False, value, ""
        valid_contracts = context.get("valid_contracts", [])
        
        if value in valid_contracts:
            return False, value, ""
            
        for valid in valid_contracts:
            norm_valid = valid.replace("I", "l").replace("0", "O")
            norm_val = value.replace("I", "l").replace("0", "O")
            if norm_valid == norm_val or value.replace("l", "I") == valid:
                return True, valid, "Matched known contract via OCR typo rules (l/I, 0/O)"
        return False, value, ""

class CompanyAliasRule:
    def apply(self, value: str, context: dict = None) -> tuple[bool, str, str]:
        if not value or not context: return False, value, ""
        company_aliases = context.get("company_aliases", {})
        upper_val = value.upper()
        
        # Ngăn match quá rộng với các chuỗi ngắn
        if len(upper_val) < 10:
            return False, value, ""
            
        for canonical_name, aliases in company_aliases.items():
            if upper_val == canonical_name:
                return False, value, ""
            for alias in aliases:
                alias_upper = alias.upper()
                
                # Case 1: alias nằm trọn trong OCR (ví dụ OCR bị dính thêm rác)
                if alias_upper in upper_val:
                    return True, canonical_name, f"Matched verified alias '{alias}'"
                    
                # Case 2: OCR nằm trong alias (bị mất đầu/cuối), nhưng phải đảm bảo đủ dài để an toàn
                if upper_val in alias_upper and len(upper_val) >= len(alias_upper) * 0.7:
                    return True, canonical_name, f"Matched verified alias '{alias}' (partial match >= 70%)"
                    
        return False, value, ""
