import typer
from pathlib import Path
import json
from rag_app.models import EvaluationDataItem
from lancedb import connect
from typing import List
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio
import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed


app = typer.Typer()


class EmbeddedEvaluationItem(BaseModel):
    question: str
    embedding: List[float]
    chunk_id: str


def batch_queries(
    queries: List[EvaluationDataItem], batch_size=20
) -> List[List[EvaluationDataItem]]:
    batch = []
    for query in queries:
        batch.append(query)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
async def embed_query(queries: List[EvaluationDataItem], client: AsyncOpenAI):
    query_strings = [query.question for query in queries]
    embeddings = await client.embeddings.create(
        input=query_strings, model="text-embedding-3-large", dimensions=256
    )
    embeddings = [embedding_object.embedding for embedding_object in embeddings.data]

    return [
        EmbeddedEvaluationItem(
            question=query.question, embedding=embedding, chunk_id=query.chunk_id
        )
        for query, embedding in zip(queries, embeddings)
    ]


async def embed_test_queries(
    queries: List[EvaluationDataItem],
) -> List[EmbeddedEvaluationItem]:
    client = AsyncOpenAI()
    batched_queries = batch_queries(queries)
    coros = [embed_query(query_batch, client) for query_batch in batched_queries]
    return await asyncio.gather(*coros)


@app.command(help="Evaluate document retrieval")
def from_json(
    input_file_path: str = typer.Option(
        help="Json file to read in labels from",
    ),
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
):
    assert Path(
        input_file_path
    ).parent.exists(), f"The directory {Path(input_file_path).parent} does not exist."
    assert (
        Path(input_file_path).suffix == ".json"
    ), "The output file must have a .json extension."
    assert Path(db_path).exists(), f"Database path {db_path} does not exist"

    with open(input_file_path, "r") as file:
        data = json.load(file)
        evaluation_data = [EvaluationDataItem(**json.loads(item)) for item in data]

    embedded_queries = asyncio.run(
        embed_test_queries(evaluation_data, db_path, table_name)
    )
