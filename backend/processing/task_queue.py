"""
Task Queue Abstraction.
For development, leverages FastAPI BackgroundTasks.
Can be easily upgraded to Celery/RQ with Redis for production environments.
"""

from fastapi import BackgroundTasks
import logging

logger = logging.getLogger(__name__)

def enqueue_document_processing(
    background_tasks: BackgroundTasks,
    document_id: str,
    file_path: str,
    metadata: dict
) -> None:
    """
    Enqueue the heavy document processing (extraction, formatting, OCR, and indexing)
    into the background queue.
    """
    from processing.document_processor import process_document_background
    
    logger.info("Enqueuing background task for document_id: %s", document_id)
    background_tasks.add_task(
        process_document_background,
        document_id=document_id,
        file_path=file_path,
        metadata=metadata
    )
