import typer
from pathlib import Path
import json
import duckdb
import os
from rag_app.models import EvaluationDataItem, KeywordExtractionResponse
from lancedb import connect
from typing import List, Union
from openai import AsyncOpenAI
import pandas as pd
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio as asyncio
from rag_app.src.chunking import batch_items
from tenacity import retry, stop_after_attempt, wait_fixed
from asyncio import run, Semaphore
from rag_app.models import TextChunk
from rag_app.src.metrics import (
    calculate_recall,
    calculate_mrr,
    slice_predictions_at_k,
)
from rich.console import Console
from collections import OrderedDict
import instructor
from rich.table import Table
import cohere
from cohere.types import RerankResponse

app = typer.Typer()
evals = OrderedDict()

client = AsyncOpenAI()
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

SIZES = [3, 5, 10, 15, 20, 25]
for size in SIZES:
    evals[f"MAR@{size}"] = slice_predictions_at_k(size, calculate_recall)
for size in SIZES:
    evals[f"MRR@{size}"] = slice_predictions_at_k(size, calculate_mrr)


class EmbeddedEvaluationItem(BaseModel):
    question: str
    embedding: List[float]
    chunk_id: str


class FullTextSearchEvaluationItem(BaseModel):
    question: str
    keywords: List[str]
    chunk_id: str


class BM25SearchEvaluationItem(BaseModel):
    question: str
    chunk_id: str


class QueryResult(BaseModel):
    source: Union[
        EmbeddedEvaluationItem, FullTextSearchEvaluationItem, BM25SearchEvaluationItem
    ]
    results: List[TextChunk]


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
async def embed_query(queries: List[EvaluationDataItem], sem: Semaphore):
    async with sem:
        try:
            query_strings = [query.question for query in queries]

            embeddings = await client.embeddings.create(
                input=query_strings, model="text-embedding-3-large", dimensions=256
            )
            embeddings = [
                embedding_object.embedding for embedding_object in embeddings.data
            ]

            return [
                EmbeddedEvaluationItem(
                    question=query.question,
                    embedding=embedding,
                    chunk_id=query.chunk_id,
                )
                for query, embedding in zip(queries, embeddings)
            ]
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e


async def embed_test_queries(
    queries: List[EvaluationDataItem],
) -> List[EmbeddedEvaluationItem]:
    batched_queries = batch_items(queries, 20)
    sem = Semaphore(10)
    coros = [embed_query(query_batch, sem) for query_batch in batched_queries]
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


async def generate_keywords_for_questions(
    queries: List[EvaluationDataItem],
) -> List[str]:
    async def generate_query_keywords(query: EvaluationDataItem, client: AsyncOpenAI):
        response: KeywordExtractionResponse = await client.chat.completions.create(
            model="gpt-4-0613",
            response_model=KeywordExtractionResponse,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class search engine. You are about to be given a question by a user. Make sure to generate as many possible keywords that are relevant to the question at hand which can help to identify relevant chunks of information to the user's query.",
                },
                {
                    "role": "assistant",
                    "content": "Make sure to extract all possible keywords within the question itself first before generating new ones. Also expand all accronyms, identify synonyms and related topics.",
                },
                {"role": "user", "content": f"The question is {query.question}."},
            ],
            max_retries=5,
        )
        return FullTextSearchEvaluationItem(
            question=query.question,
            keywords=response.keywords,
            chunk_id=query.chunk_id,
        )

    client = instructor.patch(AsyncOpenAI())
    coros = [generate_query_keywords(query, client) for query in queries]
    return await asyncio.gather(*coros)


async def match_chunks_with_keywords(
    queries: List[FullTextSearchEvaluationItem], db_path: str, table_name: str
):
    async def query_table(query: FullTextSearchEvaluationItem) -> pd.DataFrame:
        keywords = query.keywords
        keywords = [keyword.replace("'", "''") for keyword in keywords]
        keyword_conditions = " +\n".join(
            [
                f"CASE WHEN regexp_matches(text, '(?i){keyword}') THEN 1 ELSE 0 END"
                for keyword in keywords
            ]
        )
        keyword_search = "|".join(keywords)
        duckdb_query = f"""
            SELECT 
            *,
            ({keyword_conditions}) AS num_keywords_matched
        FROM 
            chunks
        WHERE 
            regexp_matches(text, '(?i)({keyword_search})')
            ORDER BY
            num_keywords_matched DESC
            LIMIT 25
        """

        db = connect(db_path)
        chunk_table = db.open_table(table_name)
        chunks = chunk_table.to_lance()
        result = duckdb.query(duckdb_query).to_df()
        return QueryResult(
            source=query,
            results=[
                TextChunk(
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "num_keywords_matched"
                    }
                )
                for index, row in result.iterrows()
            ],
        )

    coros = [query_table(query) for query in queries]
    return await asyncio.gather(*coros)


def match_chunks_with_bm25(
    db_path: str, table_name: str, queries: List[EvaluationDataItem]
):
    db = connect(db_path)
    chunk_table = db.open_table(table_name)
    try:
        chunk_table.create_fts_index("text", replace=False)
    except ValueError as e:
        print("Index on the column 'text' has already been created.")

    def query_table(query: EvaluationDataItem):
        db = connect(db_path)
        chunk_table = db.open_table(table_name)

        retrieved_queries = (
            chunk_table.search(query.question).limit(25).to_pydantic(TextChunk)
        )
        return QueryResult(
            source=BM25SearchEvaluationItem(
                question=query.question, chunk_id=query.chunk_id
            ),
            results=retrieved_queries,
        )

    return [query_table(query) for query in queries]


def score(query: QueryResult) -> dict[str, float]:
    y_true = query.source.chunk_id
    y_pred = [x.chunk_id for x in query.results]

    metrics = {label: metric_fn(y_true, y_pred) for label, metric_fn in evals.items()}
    metrics = {
        label: round(value, 2) if value != "N/A" else value
        for label, value in metrics.items()
    }
    return {
        **metrics,
        "chunk_id": query.source.chunk_id,
        "retrieved_size": len(query.results),
    }


async def rerank(results: List[QueryResult]):
    async def rerank_result(queryResult: QueryResult):
        docs = [chunk.text for chunk in queryResult.results]
        results: RerankResponse = co.rerank(
            query=queryResult.source.question,
            model="rerank-english-v2.0",
            documents=docs,
        )
        return QueryResult(
            source=queryResult.source,
            results=[queryResult.results[result.index] for result in results.results],
        )

    coros = [rerank_result(result) for result in results]
    return await asyncio.gather(*coros)


@app.command(help="Evaluate document retrieval")
def from_jsonl(
    input_file_path: str = typer.Option(
        help="Jsonl file to read in labels from",
    ),
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to read data from"),
    eval_mode=typer.Option(
        help="Query Method ( semantic,bm25 or fts )", default="semantic"
    ),
    use_reranker: bool = typer.Option(help="Use Cohere Re-Rankker", default=False),
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

    if eval_mode == "semantic":
        embedded_queries = run(embed_test_queries(evaluation_data))
        query_results = run(
            fetch_relevant_results(embedded_queries, db_path, table_name)
        )
    elif eval_mode == "fts":
        fts_queries = run(generate_keywords_for_questions(evaluation_data))
        query_results = run(
            match_chunks_with_keywords(fts_queries, db_path, table_name)
        )
    elif eval_mode == "bm25":
        query_results = match_chunks_with_bm25(db_path, table_name, evaluation_data)
    else:
        raise ValueError(
            "Invalid eval mode. Only semantic, fts or bm25 is supported at the moment"
        )

    if use_reranker:
        query_results = run(rerank(query_results))

    evals = [score(result) for result in query_results]

    df = pd.DataFrame(evals)
    df = df.set_index("chunk_id")
    console = Console()
    console.print(df)
    print("")
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    mean_values_table = Table(title="Mean Values")
    mean_values_table.add_column("Metric", style="cyan")
    mean_values_table.add_column("Value", style="magenta")
    for metric, value in numeric_df.mean().items():
        mean_values_table.add_row(metric, str(round(value, 5)))
    console.print(mean_values_table)
