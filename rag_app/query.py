import typer
import openai
from rag_app.models import TextChunk
from lancedb import connect
import textwrap
from typing import List

app = typer.Typer()


@app.command(help="Query LanceDB for some results")
def query_db(
    db_path: str,
    table_name: str,
    query: str,
):
    db = connect(db_path)
    table = db.open_table(table_name)

    client = openai.OpenAI()
    query_vector = (
        client.embeddings.create(
            input=query, model="text-embedding-3-large", dimensions=256
        )
        .data[0]
        .embedding
    )

    results: List[TextChunk] = (
        table.search(query_vector).limit(2).to_pydantic(TextChunk)
    )

    print("=========================")
    for idx, result in enumerate(results):
        print(f"Chunk {idx+1}")
        print("=========================")
        print(textwrap.fill(result.text, width=120))
        print("=========================")
