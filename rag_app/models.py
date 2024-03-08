from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    id: int
    doc_id: int
    chunk_num: int
    start_pos: int
    end_pos: int
    text: str = openai.SourceField()
    # For some reason if we call openai.ndim(), it returns 1536 instead of 256 like we want
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)
