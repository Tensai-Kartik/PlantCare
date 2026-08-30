"""
Rate Limiting & Concurrency Protection for PlantCare Backend
Protects ML inference from CPU thrashing and excessive requests on free-tier servers.
"""

import time
import asyncio
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status
from app.core.config import settings

class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.window = 60.0  # seconds
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, client_ip: str) -> bool:
        now = time.time()
        async with self._lock:
            # Clean expired timestamps
            valid_window_start = now - self.window
            self._history[client_ip] = [
                t for t in self._history[client_ip] if t > valid_window_start
            ]

            if len(self._history[client_ip]) >= self.rpm:
                return False

            self._history[client_ip].append(now)
            return True

# Concurrency semaphore for heavy ML tasks
inference_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCES)
rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
