import typer
import rag_app.query as QueryApp
import rag_app.ingest as IngestApp
import rag_app.generate_synthetic_question as GenerateApp
import rag_app.evaluate as EvaluateApp

import logfire

logfire.configure()


logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record="all"))

app = typer.Typer(
    name="Rag-App",
    help="A CLI for querying a local RAG application backed by LanceDB",
)

app.add_typer(
    QueryApp.app,
    name="query",
    help="Commands to help query your local lancedb instance",
)
app.add_typer(
    IngestApp.app,
    name="ingest",
    help="Commands to help ingest data into your local lancedb instance",
)
app.add_typer(
    GenerateApp.app,
    name="generate",
    help="Commands to help generate synthethic data from your documents",
)
app.add_typer(
    EvaluateApp.app,
    name="evaluate",
    help="Commands to help evaluate the quality of your rag application",
)
