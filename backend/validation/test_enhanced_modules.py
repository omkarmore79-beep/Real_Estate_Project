"""
Validation Tests for Enhanced Industrial Multimodal RAG Modules.

Tests the new modules:
- Enhanced chunker
- Image analyzer enhanced
- Table understanding
- Diagram understanding
- Figure understanding
- Intent detector
- Confidence scorer
- Citation builder
- Metadata enricher
- Performance optimizer
"""

from __future__ import annotations

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_chunker():
    """Test the enhanced semantic chunker."""
    logger.info("Testing enhanced chunker...")
    
    try:
        from backend.chunker_enhanced import chunk_text_pages_enhanced, CHUNK_TYPES
        
        # Test data
        pages = [
            {
                "page_number": 1,
                "text": """
                MAINTENANCE PROCEDURE
                
                1. Check hydraulic oil level
                2. Inspect hoses for leaks
                3. Replace worn seals
                
                WARNING: High pressure hazard
                """,
                "section": "Maintenance"
            }
        ]
        
        metadata = {"domain": "excavator", "machine_model": "R215L"}
        
        chunks = chunk_text_pages_enhanced(pages, "test_doc", "test.pdf", metadata)
        
        assert chunks is not None, "Chunker returned None"
        assert len(chunks) > 0, "No chunks generated"
        
        # Verify chunk has required fields
        chunk = chunks[0]
        assert "chunk_type" in chunk, "Chunk missing chunk_type"
        assert "content" in chunk, "Chunk missing content"
        
        logger.info(f"✓ Enhanced chunker test passed: {len(chunks)} chunks generated")
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced chunker test failed: {e}")
        return False


def test_image_analyzer_enhanced():
    """Test the enhanced image analyzer."""
    logger.info("Testing enhanced image analyzer...")
    
    try:
        from backend.image_analyzer_enhanced import (
            process_images_enhanced,
            classify_image_type_enhanced,
            build_multimodal_description,
        )
        
        # Test classification
        page_text = "Major Component Diagram showing boom, cab, counterweight"
        image_type, confidence = classify_image_type_enhanced(page_text, "diagram.png", is_industrial=True)
        
        assert image_type in IMAGE_TYPES, f"Unknown image type: {image_type}"
        assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
        
        # Test multimodal description
        description = build_multimodal_description(
            image_type="engineering_diagram",
            caption="Major Component Diagram",
            figure_number="3.2",
            page_text=page_text,
            ocr_text="",
            ocr_from_image=None,
            page_number=45,
            section_context="Components",
            nearby_paragraph="This diagram shows the main components",
            is_industrial=True,
        )
        
        assert description is not None, "Description is None"
        assert len(description) > 0, "Description is empty"
        
        logger.info(f"✓ Image analyzer test passed: type={image_type}, confidence={confidence:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Image analyzer test failed: {e}")
        return False


def test_table_understanding():
    """Test table understanding module."""
    logger.info("Testing table understanding...")
    
    try:
        from backend.ingestion.table_understanding import (
            extract_table_structure,
            generate_structured_table_text,
            classify_table_type,
        )
        
        # Test data
        table_data = [
            ["Reach (m)", "Height (m)", "Capacity (kg)"],
            ["4.5", "6.0", "2500"],
            ["6.0", "4.5", "1800"],
        ]
        
        structured = extract_table_structure(table_data, page_number=45, document_id="test_doc")
        
        assert structured is not None, "Table structure is None"
        assert "table_id" in structured, "Missing table_id"
        assert "column_headers" in structured, "Missing column_headers"
        
        # Test structured text
        structured_text = generate_structured_table_text(structured)
        assert structured_text is not None, "Structured text is None"
        assert "Table" in structured_text, "Structured text missing table reference"
        
        # Test classification
        table_type = classify_table_type(structured)
        assert table_type in ["lifting_chart", "specification", "pricing", "schedule", "general"]
        
        logger.info(f"✓ Table understanding test passed: type={table_type}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Table understanding test failed: {e}")
        return False


def test_diagram_understanding():
    """Test diagram understanding module."""
    logger.info("Testing diagram understanding...")
    
    try:
        from backend.ingestion.diagram_understanding import (
            generate_diagram_description,
            extract_components,
            classify_diagram_subtype,
        )
        
        # Test component extraction
        page_text = "The boom, cab, counterweight, and engine are shown"
        components = extract_components(page_text)
        
        assert isinstance(components, list), "Components is not a list"
        assert len(components) > 0, "No components extracted"
        
        # Test diagram description
        description = generate_diagram_description(
            image_type="engineering_diagram",
            caption="Major Component Diagram",
            page_text=page_text,
            figure_number="3.2",
        )
        
        assert description is not None, "Description is None"
        assert "Component" in description or "component" in description.lower(), "Description missing component reference"
        
        logger.info(f"✓ Diagram understanding test passed: {len(components)} components")
        return True
        
    except Exception as e:
        logger.error(f"✗ Diagram understanding test failed: {e}")
        return False


def test_figure_understanding():
    """Test figure understanding module."""
    logger.info("Testing figure understanding...")
    
    try:
        from backend.ingestion.figure_understanding import (
            extract_figures_from_page,
            generate_figure_title,
            generate_figure_embedding_text,
        )
        
        # Test page data
        page = {
            "page_number": 45,
            "text": "Figure 3.2: Major Component Diagram showing the boom and cab",
            "section": "Components",
            "images": [
                {"image_id": "img_001", "image_path": "/path/to/image.png"}
            ]
        }
        
        figures = extract_figures_from_page(page, "test_doc")
        
        assert isinstance(figures, list), "Figures is not a list"
        # May or may not have figures depending on detection
        
        # Test title generation
        title = generate_figure_title("3.2", "Major Component Diagram", "engineering_diagram")
        assert title is not None, "Title is None"
        assert "Figure" in title, "Title missing Figure reference"
        
        logger.info(f"✓ Figure understanding test passed: {len(figures)} figures extracted")
        return True
        
    except Exception as e:
        logger.error(f"✗ Figure understanding test failed: {e}")
        return False


def test_intent_detector():
    """Test intent detection module."""
    logger.info("Testing intent detector...")
    
    try:
        from backend.retrieval.intent_detector import (
            detect_query_intent,
            classify_query_for_routing,
            should_prioritize_images,
        )
        
        # Test image query
        query = "Show me the hydraulic diagram"
        intent = detect_query_intent(query)
        
        assert "primary_intent" in intent, "Missing primary_intent"
        assert "confidence" in intent, "Missing confidence"
        assert intent["primary_intent"] in ["image", "diagram", "table", "general"], f"Unexpected intent: {intent['primary_intent']}"
        
        # Test routing
        routing = classify_query_for_routing(query)
        assert "include_images" in routing, "Missing include_images in routing"
        
        # Test image priority
        should_prioritize = should_prioritize_images(query)
        assert isinstance(should_prioritize, bool), "should_prioritize_images not boolean"
        
        logger.info(f"✓ Intent detector test passed: intent={intent['primary_intent']}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Intent detector test failed: {e}")
        return False


def test_confidence_scorer():
    """Test confidence scoring module."""
    logger.info("Testing confidence scorer...")
    
    try:
        from backend.retrieval.confidence_scorer import (
            calculate_combined_confidence,
            score_retrieved_object,
            filter_by_confidence,
            should_hallucinate_warning,
        )
        
        # Test combined confidence
        combined = calculate_combined_confidence(
            retrieval_score=0.85,
            rerank_score=0.9,
            ocr_confidence=0.95,
            layout_confidence=0.9,
            source_type="pdf_text",
        )
        
        assert 0 <= combined <= 1, f"Invalid combined confidence: {combined}"
        
        # Test object scoring
        obj = {
            "score": 0.85,
            "rerank_score": 0.9,
            "metadata": {
                "ocr_confidence": 0.95,
                "layout_confidence": 0.9,
                "source_type": "pdf_text",
            }
        }
        scored = score_retrieved_object(obj)
        
        assert "combined_confidence" in scored, "Missing combined_confidence"
        assert "confidence_level" in scored, "Missing confidence_level"
        
        # Test hallucination warning
        should_warn, message = should_hallucinate_warning([scored])
        assert isinstance(should_warn, bool), "should_warn not boolean"
        
        logger.info(f"✓ Confidence scorer test passed: combined={combined:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Confidence scorer test failed: {e}")
        return False


def test_citation_builder():
    """Test citation builder module."""
    logger.info("Testing citation builder...")
    
    try:
        from backend.retrieval.citation_builder import (
            build_citation,
            format_citation,
            build_citations_for_results,
        )
        
        # Test citation building
        obj = {
            "id": "chunk_001",
            "score": 0.85,
            "source_type": "text",
            "metadata": {
                "document_id": "test_doc",
                "page_number": 45,
                "section": "Maintenance",
                "figure_number": "3.2",
                "source_file": "manual.pdf",
            }
        }
        
        citation = build_citation(obj)
        
        assert "document_id" in citation, "Missing document_id"
        assert "page" in citation, "Missing page"
        assert "section" in citation, "Missing section"
        
        # Test formatting
        formatted = format_citation(citation)
        assert formatted is not None, "Formatted citation is None"
        assert len(formatted) > 0, "Formatted citation is empty"
        
        logger.info(f"✓ Citation builder test passed: {formatted}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Citation builder test failed: {e}")
        return False


def test_metadata_enricher():
    """Test metadata enricher module."""
    logger.info("Testing metadata enricher...")
    
    try:
        from backend.ingestion.metadata_enricher import (
            enrich_text_chunk_metadata,
            enrich_image_metadata,
            validate_metadata_completeness,
        )
        
        # Test text chunk enrichment
        chunk = {
            "chunk_id": "chunk_001",
            "content": "Test content",
            "metadata": {
                "document_id": "test_doc",
                "page_number": 45,
            }
        }
        
        enriched = enrich_text_chunk_metadata(chunk)
        
        assert "metadata" in enriched, "Missing metadata"
        assert "ingestion_timestamp" in enriched["metadata"], "Missing ingestion_timestamp"
        
        # Test validation
        validation = validate_metadata_completeness(enriched, "text_chunk")
        assert "is_valid" in validation, "Missing is_valid in validation"
        
        logger.info(f"✓ Metadata enricher test passed: valid={validation['is_valid']}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metadata enricher test failed: {e}")
        return False


def test_performance_optimizer():
    """Test performance optimizer module."""
    logger.info("Testing performance optimizer...")
    
    try:
        from backend.ingestion.performance_optimizer import (
            detect_duplicate_chunks,
            detect_duplicate_images,
            get_document_fingerprint,
            truncate_text_for_embedding,
        )
        
        # Test duplicate detection
        chunks = [
            {"chunk_id": "1", "content": "Test content"},
            {"chunk_id": "2", "content": "Test content"},
            {"chunk_id": "3", "content": "Different content"},
        ]
        
        unique = detect_duplicate_chunks(chunks)
        assert len(unique) == 2, f"Expected 2 unique chunks, got {len(unique)}"
        
        # Test fingerprint
        doc = {"source_file": "test.pdf", "file_size": 1000, "modified_time": "2024-01-01"}
        fingerprint = get_document_fingerprint(doc)
        assert fingerprint is not None, "Fingerprint is None"
        
        # Test text truncation
        long_text = "A" * 10000
        truncated = truncate_text_for_embedding(long_text, max_length=100)
        assert len(truncated) <= 100, f"Text not truncated: {len(truncated)}"
        
        logger.info(f"✓ Performance optimizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance optimizer test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("Running Enhanced Module Validation Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced Chunker", test_enhanced_chunker),
        ("Image Analyzer Enhanced", test_image_analyzer_enhanced),
        ("Table Understanding", test_table_understanding),
        ("Diagram Understanding", test_diagram_understanding),
        ("Figure Understanding", test_figure_understanding),
        ("Intent Detector", test_intent_detector),
        ("Confidence Scorer", test_confidence_scorer),
        ("Citation Builder", test_citation_builder),
        ("Metadata Enricher", test_metadata_enricher),
        ("Performance Optimizer", test_performance_optimizer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed_count}/{total_count} tests passed")
    logger.info("=" * 60)
    
    return passed_count == total_count


if __name__ == "__main__":
    # Add backend to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import IMAGE_TYPES for test
    from backend.image_analyzer_enhanced import IMAGE_TYPES
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
