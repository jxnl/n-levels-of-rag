import frontmatter
import hashlib
from datetime import datetime
from unstructured.partition.text import partition_text
from rag_app.models import Document
from typing import Iterable

from pathlib import Path


def generate_string_hash(s: str):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_files(path: Path, file_suffix: str) -> Iterable[Document]:
    for file in path.iterdir():
        if file.suffix != file_suffix:
            continue
        post = frontmatter.load(file)
        yield Document(
            id=generate_string_hash(post.content),
            content=post.content,
            filename=file.name,
            metadata=post.metadata,
        )


def batch_chunks(chunks, batch_size=20):
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def chunk_text(
    documents: Iterable[Document], window_size: int = 1024, overlap: int = 0
):
    for doc in documents:
        for chunk_num, chunk in enumerate(partition_text(text=doc.content)):
            yield {
                "chunk_id": generate_string_hash(chunk.text),
                "chunk_number": chunk_num + 1,
                "doc_id": doc.id,
                "text": chunk.text,
                "post_title": doc.metadata.title,
                "publish_date": datetime.strptime(doc.metadata.date, "%Y-%m"),
                "source": doc.metadata.url,
            }
