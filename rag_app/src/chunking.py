import frontmatter
import hashlib
from datetime import datetime
from unstructured.partition.text import partition_text
from rag_app.models import Document
from typing import Iterable

from pathlib import Path


def read_files(path: Path, file_suffix: str) -> Iterable[Document]:
    for i, file in enumerate(path.iterdir()):
        if file.suffix != file_suffix:
            continue
        post = frontmatter.load(file)
        yield Document(
            id=hashlib.md5(post.content.encode("utf-8")).hexdigest(),
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
                "doc_id": doc.id,
                "chunk_id": chunk_num + 1,
                "text": chunk.text,
                "post_title": doc.metadata["title"],
                "publish_date": datetime.strptime(doc.metadata["date"], "%Y-%m"),
                "source": doc.metadata["url"],
            }
