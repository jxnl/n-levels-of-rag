import typer
from pathlib import Path
import json
from rag_app.models import EvaluationDataItem
from lancedb import connect
from typing import List
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio as asyncio
from rag_app.src.chunking import batch_items
from tenacity import retry, stop_after_attempt, wait_fixed
from asyncio import run
from rag_app.models import TextChunk
from sklearn.metrics import ndcg_score
import numpy as np

app = typer.Typer()


class EmbeddedEvaluationItem(BaseModel):
    question: str
    embedding: List[float]
    chunk_id: str


class QueryResult(BaseModel):
    source: EmbeddedEvaluationItem
    results: List[TextChunk]


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
    batched_queries = batch_items(queries)
    coros = [embed_query(query_batch, client) for query_batch in batched_queries]
    return await asyncio.gather(*coros)


async def fetch_relevant_results(
    queries: List[EmbeddedEvaluationItem],
    db_path: str,
    table_name: str,
) -> List[QueryResult]:
    db = connect(db_path)
    table = db.open_table(table_name)

    async def query_table(query: EmbeddedEvaluationItem):
        results = table.search(query.embedding).limit(10).to_pydantic(TextChunk)
        return QueryResult(results=results, source=query)

    coros = [query_table(query) for query in queries]
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

    embedded_queries = run(embed_test_queries(evaluation_data))
    flattened_queries = [item for sublist in embedded_queries for item in sublist]

    query_results = run(fetch_relevant_results(flattened_queries, db_path, table_name))

    for result in query_results:
        y_pred = np.linspace(1, 0, len(result.results)).tolist()

        y_true = [
            0 if item.chunk_id != result.source.chunk_id else 1
            for item in result.results
        ]
        ndcg = ndcg_score([y_true], [y_pred])

        target_chunk_id = result.source.chunk_id
        chunk_ids = [i.chunk_id for i in result.results]

        mrr = (
            0
            if target_chunk_id not in chunk_ids
            else 1 / (chunk_ids.index(target_chunk_id) + 1)
        )

        print(f"NDCG: {ndcg}, MRR: {mrr}")
