import pytest
from pipeline.adaptive_table_extractor import AdaptiveTableExtractor

def test_html_table_parser_simple():
    extractor = AdaptiveTableExtractor()
    html = """
    <table>
        <tr><td>Model</td><td>Qty</td></tr>
        <tr><td>DS-123</td><td>10PCS</td></tr>
    </table>
    """
    grid, spans, num_rows, max_cols = extractor.parse_html_table(html)
    assert num_rows == 2
    assert max_cols == 2
    assert grid[0][0]["text"] == "Model"
    assert grid[1][0]["text"] == "DS-123"
    assert grid[1][1]["text"] == "10PCS"
    assert len(spans) == 0

def test_html_table_parser_with_colspan_rowspan():
    extractor = AdaptiveTableExtractor()
    html = """
    <table>
        <tr><td colspan="2">Header Span</td></tr>
        <tr><td rowspan="2">Row Span</td><td>Detail 1</td></tr>
        <tr><td>Detail 2</td></tr>
    </table>
    """
    grid, spans, num_rows, max_cols = extractor.parse_html_table(html)
    assert num_rows == 3
    assert max_cols == 2
    # Check Header Span
    assert grid[0][0]["text"] == "Header Span"
    assert grid[0][1]["text"] == "Header Span"
    # Check Row Span
    assert grid[1][0]["text"] == "Row Span"
    assert grid[2][0]["text"] == "Row Span"
    assert grid[1][1]["text"] == "Detail 1"
    assert grid[2][1]["text"] == "Detail 2"
    assert len(spans) == 2
