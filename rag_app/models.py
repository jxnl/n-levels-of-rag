from datetime import datetime
from typing import List, Union
from pydantic import field_validator
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field

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


class Document(LanceModel):
    id: str
    content: str
    filename: str
    metadata: DocumentMetadata


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        ..., description="The reasoning process leading to the answer."
    )
    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")


class EvaluationDataItem(BaseModel):
    question: str
    answer: str
    chunk: str
    chunk_id: str
