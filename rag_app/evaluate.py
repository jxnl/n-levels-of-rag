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
from rag_app.src.metrics import (
    calculate_mrr,
    calculate_ndcg,
    slice_predictions_decorator,
)
import pandas as pd
from rich.console import Console
from collections import OrderedDict

app = typer.Typer()
evals = OrderedDict()
SIZES = [3, 5, 10, 20]
for size in SIZES:
    evals[f"MRR@{size}"] = slice_predictions_decorator(size)(calculate_mrr)

for size in SIZES:
    evals[f"NDCG@{size}"] = slice_predictions_decorator(size)(calculate_ndcg)


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
    result = await asyncio.gather(*coros)
    return [item for sublist in result for item in sublist]


async def fetch_relevant_results(
    queries: List[EmbeddedEvaluationItem],
    db_path: str,
    table_name: str,
) -> List[QueryResult]:
    db = connect(db_path)
    table = db.open_table(table_name)

    async def query_table(query: EmbeddedEvaluationItem):
        results = table.search(query.embedding).limit(25).to_pydantic(TextChunk)
        return QueryResult(results=results, source=query)

    coros = [query_table(query) for query in queries]
    return await asyncio.gather(*coros)


def score(query: QueryResult) -> dict[str, float]:
    y_true = query.source.chunk_id
    y_pred = [x.chunk_id for x in query.results]

    metrics = {label: metric_fn(y_true, y_pred) for label, metric_fn in evals.items()}
    return {**metrics, "chunk_id": query.source.chunk_id}


@app.command(help="Evaluate document retrieval")
def from_jsonl(
    input_file_path: str = typer.Option(
        help="Jsonl file to read in labels from",
    ),
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to read data from"),
):
    assert Path(
        input_file_path
    ).parent.exists(), f"The directory {Path(input_file_path).parent} does not exist."
    assert (
        Path(input_file_path).suffix == ".jsonl"
    ), "The output file must have a .jsonl extension."
    assert Path(db_path).exists(), f"Database path {db_path} does not exist"

    with open(input_file_path, "r") as file:
        data = file.readlines()
        evaluation_data = [EvaluationDataItem(**json.loads(item)) for item in data]

    embedded_queries = run(embed_test_queries(evaluation_data))

    query_results = run(fetch_relevant_results(embedded_queries, db_path, table_name))

    evals = [score(result) for result in query_results]

    df = pd.DataFrame(evals)
    df = df.set_index("chunk_id")
    console = Console()
    console.print(df)
