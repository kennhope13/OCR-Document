import re
from typing import List, Union, Dict, Any

def normalize_text(text: str) -> str:
    """Normalizes text by converting to lowercase and stripping extra whitespaces."""
    if text is None:
        return ""
    text = str(text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def calculate_edit_operations(ref: Union[str, List[str]], hyp: Union[str, List[str]]) -> Dict[str, int]:
    """
    Calculates the minimum edit operations (insertions, deletions, substitutions)
    to transform ref into hyp using dynamic programming.
    Works for both strings (CER) and lists of words (WER).
    """
    n = len(ref)
    m = len(hyp)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion from ref
                dp[i][j-1] + 1,      # insertion into ref
                dp[i-1][j-1] + cost  # substitution
            )
            
    # Backtrack to find number of I, D, S
    i, j = n, m
    insertions = 0
    deletions = 0
    substitutions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1
                
    return {
        "distance": dp[n][m],
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "ref_length": n
    }

def evaluate_text_pair(expected: str, actual: str) -> Dict[str, Any]:
    """
    Evaluates a single pair of expected (ground truth) and actual (OCR) strings.
    Returns CER, WER, and exact match metrics.
    """
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    
    # CER
    cer_edits = calculate_edit_operations(expected_norm, actual_norm)
    cer = cer_edits["distance"] / max(1, cer_edits["ref_length"])
    
    # WER
    expected_words = expected_norm.split()
    actual_words = actual_norm.split()
    wer_edits = calculate_edit_operations(expected_words, actual_words)
    wer = wer_edits["distance"] / max(1, wer_edits["ref_length"])
    
    # Exact Match
    strict_match = (expected == actual) if (expected is not None and actual is not None) else False
    normalized_match = (expected_norm == actual_norm)
    
    return {
        "expected": expected,
        "actual": actual,
        "expected_norm": expected_norm,
        "actual_norm": actual_norm,
        "cer": cer,
        "wer": wer,
        "cer_edits": cer_edits,
        "wer_edits": wer_edits,
        "strict_match": strict_match,
        "normalized_match": normalized_match
    }
