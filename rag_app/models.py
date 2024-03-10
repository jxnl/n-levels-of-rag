from datetime import datetime
from typing import List, Union
from pydantic import field_validator
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
    source: str


class Document(BaseModel):
    id: str
    content: str
    filename: str
    metadata: dict[str, Union[str, List[str]]]

    @field_validator('metadata')
    @classmethod
    def metadata_must_contain_a_valid_datestring(cls, v: dict[str, Union[str, List[str]]]):
        try:
            datetime.strptime(v["date"], "%Y-%m")
        except Exception as e:
            raise ValueError(
                f"Date format must be YYYY-MM (Eg. 2024-10). Unable to parse provided date of {v['date']} "
            )

        return v
    
    @field_validator('metadata')
    @classmethod
    def metadata_must_contain_required_keys(cls,v:dict[str, Union[str, List[str]]]):
        required_keys = [
            "url","date","title"
        ]

        for k in required_keys:
            if k not in v:
                raise ValueError(f"Required Property {k} is not present in metadata")
        return v
