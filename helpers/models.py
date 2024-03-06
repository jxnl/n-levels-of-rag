from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

openai = get_registry().get("openai").create(model="text-embedding-3-large", ndims=256)


class Document(BaseModel):
    id: int
    text: str
    filename: str


class TextChunk(LanceModel):
    id: int
    doc_id: int
    chunk_num: int
    start_pos: int
    end_pos: int
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)
