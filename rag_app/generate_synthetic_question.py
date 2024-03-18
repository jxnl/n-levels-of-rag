
import typer
from pathlib import Path
from rag_app.src.chunking import read_files, chunk_text
from pydantic import BaseModel, Field
from instructor import patch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio as asyncio
from asyncio import run
from rag_app.models import TextChunk
import json
from pathlib import Path
from typing import List

import tqdm
import typer
from instructor import patch
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from rag_app.models import TextChunk, EvaluationDataItem, QuestionAnswerPair
from rag_app.src.chunking import chunk_text, read_files

app = typer.Typer()


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
async def generate_question_answer_pair(
    client: AsyncOpenAI,
    chunk: TextChunk,
) -> tuple[QuestionAnswerPair, TextChunk]:
    res = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a world class algorithm that excels at generating great questions that can be only answered by a specific text that will soon be passed to you. ",
            },
            {
                "role": "assistant",
                "content": f"Generate a question and answer pair that uses information and content that is specific to the following text chunk, including a chain of thought:\n\n{chunk}",
            },
        ],
        response_model=QuestionAnswerPair,
    )
    return (res, chunk)


async def gather_questions(
    chunks: TextChunk,
) -> List[EvaluationDataItem]:
    client = patch(AsyncOpenAI())
    coros = [generate_question_answer_pair(client, chunk) for chunk in chunks]
    output = []
    for response in asyncio.as_completed(coros):
        questionAnswer, chunkData = await response
        assert isinstance(chunkData, TextChunk)
        assert isinstance(questionAnswer, QuestionAnswerPair)
        output.append(
            EvaluationDataItem(
                **{
                    "question": questionAnswer.question,
                    "answer": questionAnswer.answer,
                    "chunk": chunkData.text,
                    "chunk_id": chunkData.chunk_id,
                }
            )
        )
    return output


@app.command(help="Generate questions for each chunk in a given file")
def synthethic_questions(
    folder_path: str = typer.Option(help="Folder to read data from"),
    max_questions: int = typer.Option(
        help="max number of question/answer pairs to generate", default=-1
    ),
    output_path: str = typer.Option(
        help="Json file to write output to", default="output.json"
    ),
):
    assert Path(
        output_path
    ).parent.exists(), f"The directory {Path(output_path).parent} does not exist."
    assert (
        Path(output_path).suffix == ".json"
    ), "The output file must have a .json extension."

    file = read_files(Path(folder_path), file_suffix=".md")
    chunks = chunk_text(file)
    chunks = [TextChunk(**chunk) for chunk in chunks]
    if max_questions > 0:
        chunks = chunks[:max_questions]

    output = run(gather_questions(chunks))
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
