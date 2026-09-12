"""Hash adapter for the explicitly reused simulator."""
import hashlib
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
