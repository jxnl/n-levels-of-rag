from datetime import datetime
from typing import List, Union
from pydantic import field_validator
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    chunk_id: str
    doc_id: str
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)
    post_title: str
    publish_date: datetime
    chunk_number: int
    source: str


class DocumentMetadata(LanceModel):
    date: str
    url: str
    title: str

    @field_validator("date")
    @classmethod
    def metadata_must_contain_a_valid_date_string(cls, v: str):
        try:
            datetime.strptime(v, "%Y-%m")
        except Exception as e:
            raise ValueError(
                f"Date format must be YYYY-MM (Eg. 2024-10). Unable to parse provided date of {v} "
            )
        return v


class Document(BaseModel):
    id: str
    content: str
    filename: str
    metadata: DocumentMetadata
