"""
Conversational Session Memory Manager.
Tracks messages by session_id in a thread-safe manner.
"""

from __future__ import annotations

import threading
from typing import Any

class SessionMemory:
    """Stores conversation history for a single chat session."""
    
    def __init__(self, max_history_pairs: int = 10):
        self.history: list[dict[str, str]] = []
        self.max_history_pairs = max_history_pairs

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Each pair has a user message and assistant response
        if len(self.history) > self.max_history_pairs * 2:
            self.history = self.history[-self.max_history_pairs * 2:]

    def get_history(self) -> list[dict[str, str]]:
        return list(self.history)

    def clear(self):
        self.history = []


class ConversationalMemoryManager:
    """Thread-safe manager for multiple session memories."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMemory] = {}

    def get_session(self, session_id: str) -> SessionMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory()
            return self._sessions[session_id]

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()


# Global Singleton Manager
memory_manager = ConversationalMemoryManager()
