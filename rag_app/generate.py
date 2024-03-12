import typer
from pathlib import Path
from rag_app.src.chunking import read_files, chunk_text
from pydantic import BaseModel, Field
from instructor import patch
from openai import OpenAI
from tqdm import tqdm

app = typer.Typer()


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")
    chain_of_thought: str = Field(
        ..., description="The reasoning process leading to the answer."
    )


client = patch(OpenAI())


def generate_question_answer_pair(chunk: str) -> QuestionAnswerPair:
    question_answer = client.chat.completions.create(
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
    return question_answer


@app.command(help="Generate questions for each chunk in a given file")
def questions(
    folder_path: str = typer.Option(help="Folder to read data from"),
    max_questions: int = typer.Option(
        help="max number of question/answer pairs to generate", default=10
    ),
):
    file = read_files(Path(folder_path), file_suffix=".md")
    chunks = chunk_text(file)

    import json

    output = []
    for idx, chunk in tqdm(enumerate(chunks), total=max_questions):
        if idx == max_questions:
            break
        response = generate_question_answer_pair(chunk["text"])
        output.append(
            {
                "question": response.question,
                "answer": response.answer,
                "chunk": chunk["text"],
            }
        )
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)
