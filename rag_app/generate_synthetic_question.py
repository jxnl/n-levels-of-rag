import typer
from pathlib import Path
from rag_app.src.chunking import read_files, chunk_text
from pydantic import BaseModel, Field
from instructor import patch
from openai import AsyncOpenAI
import tqdm
import asyncio
from rag_app.models import TextChunk
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List

app = typer.Typer()


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


client = patch(AsyncOpenAI())


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
async def generate_question_answer_pair(
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


async def gather_questions(chunks) -> List[tuple[QuestionAnswerPair, TextChunk]]:
    coros = [generate_question_answer_pair(chunk) for chunk in chunks]
    output = []
    for response in tqdm.asyncio.tqdm_asyncio.as_completed(coros):
        questionAnswer, chunkData = await response
        assert isinstance(chunkData, TextChunk)
        assert isinstance(questionAnswer, QuestionAnswerPair)
        output.append(
            {
                "question": questionAnswer.question,
                "answer": questionAnswer.answer,
                "chunk": chunkData.text,
                "chunk_id": chunkData.chunk_id,
            }
        )
    return output


@app.command(help="Generate questions for each chunk in a given file")
def synthethic_questions(
    folder_path: str = typer.Option(help="Folder to read data from"),
    max_questions: int = typer.Option(
        help="max number of question/answer pairs to generate", default=-1
    ),
    output_path: str = typer.Option(
        help="File Path to write output to", default="output.json"
    ),
):
    file = read_files(Path(folder_path), file_suffix=".md")
    chunks = chunk_text(file)
    chunks = [TextChunk(**chunk) for chunk in chunks]
    if max_questions > 0:
        chunks = chunks[:max_questions]

    output = asyncio.run(gather_questions(chunks))
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
