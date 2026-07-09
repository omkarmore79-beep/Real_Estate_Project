"""
Consolidated Status Wrapper.
Delegates to backend/storage/doc_status.py.
"""

from storage.doc_status import (
    set_status as update_status,
    get_status,
    is_ready,
    is_processing,
    recover_interrupted_tasks
)

def save_initial_status(document_id: str, status: str = "uploaded", progress: int = 5, filename: str = "") -> None:
    """Save the initial status record on file receipt."""
    update_status(
        document_id=document_id,
        status=status,
        message="Document received. Starting processing."
    )
