from pathlib import Path
from typing import Protocol

from docflow.schema import ExtractedDocument


class Extractor(Protocol):
    name: str

    def can_handle(self, path: Path) -> bool: ...

    def extract(self, path: Path) -> ExtractedDocument: ...
