from datetime import datetime
from typing import List, Union

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    doc_id: str
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)
    post_title: str
    publish_date: datetime
    chunk_id: int
    post_chunk_count: int
    source: str


class Document(BaseModel):
    id: str
    content: str
    filename: str
    metadata: dict[str, Union[str, List[str]]]
