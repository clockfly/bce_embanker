import time
import functools
import asyncio
from typing import Callable, Any

def get_time(func: Callable) -> Callable:
    """Decorator to measure execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def get_time_async(func: Callable) -> Callable:
    """Decorator to measure execution time of an async function"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper