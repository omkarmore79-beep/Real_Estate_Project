"""
Table Understanding Module for Industrial Multimodal RAG.

Converts tables into structured data instead of flattened text.
Preserves:
- Title
- Table number
- Column headers
- Row headers
- Units
- Values
- Merged cells

Generates structured table text for embedding.
Creates separate vector for every table.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_table_structure(table_data: list[list[str]], page_number: int, document_id: str) -> dict:
    """
    Extract structured information from a table.
    
    Args:
        table_data: 2D list of table cells (from PyMuPDF table extraction)
        page_number: Page number where table is located
        document_id: Document identifier
    
    Returns:
        Structured table dictionary with metadata and content
    """
    if not table_data or len(table_data) == 0:
        return {}
    
    # Detect header row (usually first row with most text)
    header_row = table_data[0] if table_data else []
    
    # Detect if first column contains row headers (e.g., "Height", "Reach", "Arm")
    has_row_headers = False
    if len(table_data) > 1 and len(table_data[0]) > 1:
        # Check if first column values look like labels rather than data
        first_col_values = [row[0] for row in table_data[1:] if row and len(row) > 0]
        if first_col_values:
            # Row headers are typically text, not numbers
            text_count = sum(1 for v in first_col_values if re.search(r'[a-zA-Z]', str(v)))
            has_row_headers = text_count / len(first_col_values) > 0.5
    
    # Extract column headers
    column_headers = [str(h).strip() for h in header_row] if header_row else []
    
    # Extract row headers and data
    row_headers = []
    data_rows = []
    
    for row in table_data[1:]:
        if not row:
            continue
        
        if has_row_headers and len(row) > 0:
            row_headers.append(str(row[0]).strip())
            data_rows.append([str(cell).strip() for cell in row[1:]])
        else:
            data_rows.append([str(cell).strip() for cell in row])
    
    # Extract units from headers or data
    units = extract_units(column_headers, data_rows)
    
    # Detect table number from context
    table_number = detect_table_number(header_row, page_number)
    
    # Generate table title
    table_title = generate_table_title(column_headers, row_headers, table_number)
    
    # Build structured table
    table_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_table_{page_number}_{table_number}"))
    
    structured_table = {
        "table_id": table_id,
        "document_id": document_id,
        "page_number": page_number,
        "table_number": table_number,
        "title": table_title,
        "column_headers": column_headers,
        "row_headers": row_headers,
        "data_rows": data_rows,
        "units": units,
        "row_count": len(data_rows),
        "column_count": len(column_headers) if column_headers else 0,
        "has_row_headers": has_row_headers,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return structured_table


def extract_units(column_headers: list[str], data_rows: list[list[str]]) -> dict[str, str]:
    """
    Extract units from column headers or data.
    
    Returns a mapping of column index to unit (e.g., "m", "kg", "ft").
    """
    units = {}
    unit_patterns = [
        r'\b(m|mm|cm|km|ft|in|inch|meter|metre)\b',
        r'\b(kg|g|lb|pound|ton|tonne)\b',
        r'\b(N|Nm|kN|MPa|PSI|bar|pascal)\b',
        r'\b(L|l|gal|gallon|liter|litre)\b',
        r'\b(deg|°|degree|radian)\b',
        r'\b(%|percent|percentage)\b',
        r'\b(RPM|rpm|rev|min)\b',
        r'\b(V|volt|amp|watt|kW|MW)\b',
    ]
    
    for col_idx, header in enumerate(column_headers):
        header_lower = header.lower()
        for pattern in unit_patterns:
            match = re.search(pattern, header_lower, re.IGNORECASE)
            if match:
                units[col_idx] = match.group(0)
                break
    
    return units


def detect_table_number(header_row: list[str], page_number: int) -> str:
    """
    Detect table number from header row or generate one.
    
    Looks for patterns like "Table 3.2", "Table A1", etc.
    """
    header_text = " ".join(header_row) if header_row else ""
    
    # Try to extract table number
    table_match = re.search(r'(?:table|tbl)\s*[\#\-]?\s*([A-Za-z0-9\.\-]+)', header_text, re.IGNORECASE)
    if table_match:
        return table_match.group(1)
    
    # Generate default table number
    return f"T{page_number}"


def generate_table_title(column_headers: list[str], row_headers: list[str], table_number: str) -> str:
    """
    Generate a descriptive title for the table.
    
    Uses column headers and row headers to create a meaningful title.
    """
    if not column_headers:
        return f"Table {table_number}"
    
    # Use first few column headers to create title
    title_parts = [h for h in column_headers[:3] if h and len(h) > 2]
    
    if row_headers:
        # Add context about what's being measured
        first_row_header = row_headers[0] if row_headers else ""
        if first_row_header and len(first_row_header) > 2:
            title_parts.insert(0, first_row_header)
    
    if title_parts:
        title = " vs ".join(title_parts[:3])
        return f"Table {table_number}: {title}"
    
    return f"Table {table_number}"


def generate_structured_table_text(structured_table: dict) -> str:
    """
    Generate structured text description for table embedding.
    
    This text preserves the relational structure instead of flattening.
    """
    parts = []
    
    # Table identification
    table_number = structured_table.get("table_number", "")
    title = structured_table.get("title", "")
    if table_number:
        parts.append(f"Table {table_number}")
    if title:
        parts.append(title)
    
    # Column headers with units
    column_headers = structured_table.get("column_headers", [])
    units = structured_table.get("units", {})
    
    if column_headers:
        header_text = "Columns: " + ", ".join(column_headers)
        if units:
            unit_text = ", ".join([f"{col_headers[i]} ({units[i]})" for i in units if i < len(column_headers)])
            if unit_text:
                header_text = f"Columns with units: {unit_text}"
        parts.append(header_text)
    
    # Row headers
    row_headers = structured_table.get("row_headers", [])
    if row_headers:
        parts.append(f"Row categories: {', '.join(row_headers[:5])}")
    
    # Sample data rows (first 3)
    data_rows = structured_table.get("data_rows", [])
    if data_rows:
        sample_count = min(3, len(data_rows))
        parts.append(f"Data rows ({sample_count} shown):")
        
        for i in range(sample_count):
            row = data_rows[i]
            if row_headers and i < len(row_headers):
                row_label = row_headers[i]
                row_text = f"  {row_label}: {', '.join(row[:4])}"
            else:
                row_text = f"  Row {i+1}: {', '.join(row[:4])}"
            parts.append(row_text)
    
    # Table statistics
    row_count = structured_table.get("row_count", 0)
    column_count = structured_table.get("column_count", 0)
    parts.append(f"Total: {row_count} rows × {column_count} columns")
    
    # Combine all parts
    structured_text = ". ".join(parts)
    
    # Limit length for embedding
    return structured_text[:1200]


def extract_table_from_pdf_page(page) -> list[dict]:
    """
    Extract all tables from a PDF page using PyMuPDF.
    
    Returns list of structured table dictionaries.
    """
    tables = []
    
    try:
        if hasattr(page, "find_tables"):
            table_finder = page.find_tables()
            if table_finder and table_finder.tables:
                for table_idx, table in enumerate(table_finder.tables):
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 0:
                        # Get table bbox for reference
                        bbox = table.bbox
                        
                        # Convert to structured format
                        structured = {
                            "table_index": table_idx,
                            "bbox": bbox,
                            "raw_data": table_data,
                        }
                        tables.append(structured)
    except Exception as exc:
        logger.warning("Table extraction failed on page: %s", exc)
    
    return tables


def process_tables_from_pages(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Process all tables from PDF pages into structured format.
    
    Args:
        pages: List of page dictionaries from PDF processor
        document_id: Document identifier
        metadata: Additional metadata to include
    
    Returns:
        List of structured table records for indexing
    """
    meta = metadata or {}
    table_records = []
    
    for page in pages:
        page_number = page.get("page_number", 0)
        
        # Extract tables from page
        tables = page.get("tables", [])
        
        for table_idx, table_info in enumerate(tables):
            table_data = table_info.get("raw_data", [])
            
            if not table_data:
                continue
            
            # Convert to structured format
            structured_table = extract_table_structure(
                table_data, page_number, document_id
            )
            
            if not structured_table:
                continue
            
            # Generate structured text for embedding
            structured_text = generate_structured_table_text(structured_table)
            
            # Build table record
            table_record = {
                "table_id": structured_table["table_id"],
                "document_id": document_id,
                "page_number": page_number,
                "table_number": structured_table["table_number"],
                "title": structured_table["title"],
                "structured_text": structured_text,
                "vector": [],  # filled by embedding service
                "metadata": {
                    "document_id": document_id,
                    "page_number": page_number,
                    "table_id": structured_table["table_id"],
                    "table_number": structured_table["table_number"],
                    "title": structured_table["title"],
                    "column_headers": structured_table["column_headers"],
                    "row_headers": structured_table["row_headers"],
                    "row_count": structured_table["row_count"],
                    "column_count": structured_table["column_count"],
                    "units": structured_table["units"],
                    "source_file": meta.get("source_file", ""),
                    "project": meta.get("project_name", meta.get("project", "")),
                    "builder": meta.get("builder", ""),
                    "document_type": meta.get("document_type", ""),
                    "domain": meta.get("domain", "generic"),
                    "machine_model": meta.get("machine_model", ""),
                    "ingestion_timestamp": structured_table["ingestion_timestamp"],
                },
            }
            
            table_records.append(table_record)
    
    logger.info(
        "Processed %d structured tables for document_id=%s",
        len(table_records), document_id
    )
    
    return table_records


def is_lifting_chart(structured_table: dict) -> bool:
    """
    Detect if a table is a lifting chart (common in excavator manuals).
    
    Lifting charts typically have columns like:
    - Reach/Radius
    - Height
    - Capacity/Load
    - Boom length
    """
    column_headers = [h.lower() for h in structured_table.get("column_headers", [])]
    
    lifting_keywords = [
        "reach", "radius", "height", "capacity", "load", "boom", "arm",
        "lifting", "working", "range", "weight"
    ]
    
    keyword_count = sum(1 for keyword in lifting_keywords if any(keyword in h for h in column_headers))
    
    return keyword_count >= 2


def is_specification_table(structured_table: dict) -> bool:
    """
    Detect if a table contains technical specifications.
    
    Spec tables typically have:
    - Parameter names in first column
    - Values in second column
    - Units in headers
    """
    row_headers = structured_table.get("row_headers", [])
    column_headers = [h.lower() for h in structured_table.get("column_headers", [])]
    
    # Check for spec-like column headers
    spec_headers = ["parameter", "spec", "specification", "item", "description", "value"]
    has_spec_header = any(any(sh in h for h in column_headers) for sh in spec_headers)
    
    # Check for units in headers
    has_units = bool(structured_table.get("units"))
    
    return has_spec_header or (has_units and len(row_headers) > 0)


def classify_table_type(structured_table: dict) -> str:
    """
    Classify the type of table for better retrieval.
    
    Returns: lifting_chart, specification, pricing, schedule, general
    """
    if is_lifting_chart(structured_table):
        return "lifting_chart"
    
    if is_specification_table(structured_table):
        return "specification"
    
    column_headers = [h.lower() for h in structured_table.get("column_headers", [])]
    
    # Check for pricing/payment tables
    pricing_keywords = ["price", "cost", "payment", "installment", "amount", "rate"]
    if any(any(pk in h for h in column_headers) for pk in pricing_keywords):
        return "pricing"
    
    # Check for schedule/timeline tables
    schedule_keywords = ["date", "time", "schedule", "milestone", "phase", "stage"]
    if any(any(sk in h for h in column_headers) for sk in schedule_keywords):
        return "schedule"
    
    return "general"
