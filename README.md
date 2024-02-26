# Levels of Complexity: RAG Applications

The goal of this repository is to go over different levels of complexity with an RAG application. These are my loose notes as I build out this repository.

## Level 1: The Basics

1. Recursively traverses file system to generate text.
2. Simple generator to chunk text.
3. Generator that batches requests and sends it to an embedding API using asyncio
4. Save data to LanceDB
5. CLI to ask a question, embed question, yield chunks and generate response

### Processing Pipeline

```python
from dataclasses import dataclass
from typing import Iterable, List
import asyncio

sem = asyncio.Semaphore(10)

class TextChunk(BaseModel):
    id: int
    text: str
    embedding: np.array
    filename: str
    uuid: str = Field(default_factory=uuid.uuid4)

def flatmap(f, items):
    for item in items:
        for subitem in f(item):
            yield subitem

def get_texts():
    for file in files:
        yield TextChunk(
            text=file.read(),
            embedding=None,
            filename=file.name
        )

def chunk_text(items:Iterable[TextChunk], window_size: int, overlap: int=0):
    for i in range(0, len(items), window_size-overlap):
        yield TextChunk(
            text = items[i:i+window_size],
            embedding = None,
            filename = items[i].filename
        )

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def embed_batch(chunks: List[TextChunk]) -> List[TextChunk]:
    texts = [chunk.text for chunk in chunks]
    resp = embedding_api( # this is just the openai call
        texts=texts
    )
    for chunk, embedding in zip(chunks, resp):
        chunk.embedding = embedding
        yield chunks

def save_batch(chunks: List[TextChunk]):
    for chunk in chunks:
        db.insert(chunk)

if __name__ == "__main__":
    # This is the CLI
    texts = get_texts()
    chunks = flatmap(chunk_text, texts)
    batched_chunks = batched(chunks, 10)
    for chunks in tqdm(batched_chunks):
        chunk_with_embedding = embed_batch(chunks)
        save_batch(chunk_with_embedding)
```

### Search Pipeline

```python
def search(question: str) -> List[TextChunk]:
    embeddings = embedding_api(texts=[question])
    results = db.search(question)
    return results

if __name__ == "__main__":
    question = input("Ask a question: ")
    results = search(question)
    for chunk in results:
        print(chunk.text)
```

### Answer Pipeline

```python
def answer(question: str, results: List[TextChunk]) -> str:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        stream=False,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt(question, results)}
        ]
    )

if __name__ == "__main__":
    question = input("Ask a question: ")
    results = search(question)
    response = answer(question, results)
    for chunk in response:
        print(chunk.text, end="")
```

## Level 2 Structure RAG Application

Step 2 is introducing more structure in the inputs and the outputs of your model, allowing for some more complexity in the entire pipeline.

### Processing

1. Better Asyncio
2. Better Chunking
3. Better Retries

### Search

1. Better Ranking (Cohere)
2. Query Expansion / Rewriting
3. Parallel Queries

```python
class SearchQuery(BaseModel):
    semantic_search: str

def extract_query(question: str) -> Iterable[SearchQuery]:
    return client.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract a query"
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_model=Iterable[SearchQuery]
    )

def search(search: Iterable[SearchQuery]) -> List[TextChunk]:
    with LanceDB() as db:
        results = db.search(search)
        return results


def rerank(question: str, results: List[TextChunk]) -> List[TextChunk]:
    return cohere_api(
        question=question,
        texts=[chunk.text for chunk in results]
    )

if __name__ == "__main__":
    question = input("Ask a question: ")
    search_query = extract_query(question)
    results = search(search_query)
    ranked_results = rerank(question, results)
    for chunk in ranked_results:
        print(chunk.text)
```

### Answering

1. Citating specific text chunks
2. Streaming Response Model for better structure.

```python
class MultiPartResponse(BaseModel):
    response: str
    followups: List[str]
    sources: List[int]

def answer(question: str, results: List[TextChunk]) -> Iterable[MultiPartResponse]:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        stream=True,
        response_model=instructor.Partial[MultiPartResponse]
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt(question, results)}
        ]
    )

if __name__ == "__main__":
    from rich.console import Console

    question = input("Ask a question: ")

    search_query = extract_query(question)
    results = search(search_query)
    results = rerank(question, results)
    response = answer(question, results)

    console = Console()
    for chunk in response:
        console.clear()
        console.print(chunk.dump_model_json())
```

## Level 3: Observability

Level 3 is when you should start thinking about observing and measuring the different aspects of your application. You want to have various logs that can help you identify bottlenecks and issues in your application. Here are some logs that you'll find very useful.

### Log how the queries are being rewritten

1. When addressing a complaint we should quickly understand if the query was written correctly

| Query | Rewritten Query | latency | ... |
| ----- | --------------- | ------- | --- |
| ...   | ...             | ...     | ... |

example: once we found that for queries with "latest" the dates it was selecting was literally the current date, we were able to quickly fix the issue by including few shot examples that consider latest to be 1 week or more.

1. Training a model

We can also use all the positive examples to figure out how to train a model that does query expansion better.

### Log the citations

By logging the citations, we can quickly understand if the model is citing the correct information, what text chunks are popular, review and understand if the model is citing the correct information. and also potentially build a model in the future that can understand what text chunks are more important.

| Query | Rewritten Query | ... | Citations |
| ----- | --------------- | --- | --------- |
| ...   | ...             | ... | [1,2,4]   |

There's a couple ways you can do this. For example, when you cite something, you can include not only what was shown to the language model, but also what was cited. If something was shown as a language model but was not cited, we can include this as part of the dataset.

| Query | Rewritten Query | ... | sources | cited |
| ----- | --------------- | --- | ------- | ----- |
| ...   | ...             | ... | [1,2,4] | [1,2] |

### Log mean cosine scores and reranker scores

By attaching this little metadata, we will be able to very cheaply identify queries that may be performing poorly.

| Query                          | Rewritten Query                | ... | Mean Cosine Score | Reranker Score |
| ------------------------------ | ------------------------------ | --- | ----------------- | -------------- |
| What is the capital of France? | What is the capital of France? | ... | 0.9               | 0.8            |
| Who modified the file last?    | Who modified the file last?    | ... | 0.2               | 0.1            |

Here you might see "oh clearly i can't answer questions about file modification" thats not even in my index.

### Log user level metadata for the search

By including other group information information, we can quickly identify if a certain group is having a bad experience

Examples could be

1. Organization ID
2. User ID
3. User Role
4. Signup Date
5. Device Type
6. Geo Location
7. Language

This could help you understand a lot of different things about how your application is being used. Maybe people on a different device are asking shorter queries and they are performing poorly. Or maybe when a new organization signed up, the types of questions they were asking were being served poorly by the language model. In the future we'll talk about other metrics, but just by implementing the mean cosine score and the free ranker score, you get these things for free without any additional work.

Just by building up some simple dashboards that are grouped by these attributes and look at the average scores, you can learn a lot. My recommendation is to review these things during stand-up once a week, look at some examples, and figure out what we could do to improve our system. When we see poor scores, we can look at the query and the rewritten query and try to understand what exactly is going on.

### Have Users

By this point you should definitely be having users. You've already set yourself for success by understanding queries, rewriting them, and monitoring how users are actually using your system. The next couple of steps will be around improving specific metrics and also different ways of doing that.

## Level 4: Evals

I think for the most part there are two separate systems, the search system and the question answering system. Many people evaluate the question answering system without evaluating the search system.

### Evaluating the Search System

The goal here should be focusing on things like precision and recall at K. Since you've already logged all the citation data, you can easily use a language model to help you evaluate the search system.

If you don't have enough data for that, you can also always use synthetic data. We can take random text chunks or random documents. I ask the language model to generate questions and at least verify that the search system can identify the text chunks that would generate these questions.

```python

def test():
    text_chunk = sample_text_chunk()
    questions = ask_ai(f"generate questions that could be ansered by {text_chunk.text}")
    for question in questions:
        search_results = search(question)

    return {
        "recall@5": (1 if text_chunk in search_results[:5] else 0),
        ...
    }

average_recall = sum(test() for _ in range(n)) / n
```

Your code shouldn't actually look like this, but this generally captures the idea that we can synthetically generate questions and use them as part of our evaluation. You can try to be creative. But ultimately it will be a function of how well you can actually write a generation prompt.

### Evaluating the Answering System

This is a lot trickier, but often times people will use a framework like, I guess, to evaluate the questions. Here I recommend spending some more time building out a data set that actually has answers.

```python
def test():
    text_chunk = sample_text_chunks(n=...)
    question, answer = ask_ai(f"generate questions and answers for {text_chunk.text}")

    ai_answer = rag_app(question)
    return ask_ai(f"for the question {question} is the answer {ai_answer} correct given that {answer} is the correct answer?")
```

#### Evaluating the Answering System: Feedback

It's also good to build in feedback mechanisms in order to get better scores. I recommend building a thumbs up, thumbs down rating system rather than a five star rating system. I won't go into details right now, but this is something I strongly recommend.

### The purpose of synethic data

The purpose of synthetic data is to help you quickly get some metrics out. It will help you build out this evaluation pipeline in hopes that as you get more users and more real questions, you'll be able to understand where we're performing well and where we're performing poorly using the suite of tests that we have. Precision, recall, mean ranking scores, etc.

## Level 5: Understanding Short comings

At this point you should be able to have a data set that is extremely diverse using both the synthetic data and production data. We should also have a suite of scores that we can use to evaluate the quality of our answers.

| org_id | query | rewritten | answer | recall@k | precision@k | mean ranking score | reranker score | user feedback | citations | sources | ... |
| ------ | ----- | --------- | ------ | -------- | ----------- | ------------------ | -------------- | ------------- | --------- | ------- | --- |
| org123 | ...   | ...       | ...    | ...      | ...         | ...                | ...            | ...           | ...       | ...     | ... |

Now we can do a bunch of different things to understand how we're doing by doing exploratory data analysis. We can look at the mean ranking score and reranker score and see if there are any patterns. We can look at the citations and see if there are any patterns. We can look at the user feedback and see if there are any patterns. We can look at the sources and see if there are any patterns. We can look at the queries and see if there are any patterns.

### Clustering Queries

We can use clustering to understand if there are any patterns in the queries. We can use clustering to understand if there are any patterns in the citations. We can use clustering to understand if there are any patterns in the sources. We can use clustering to understand if there are any patterns in the user feedback.

We'll go into more depth later, but the general idea is we can also introduce cluster topics. I find that there's usually two different kinds of clutches that we detect.

1. Topics
2. Capabilities

Topics are captured by the nature of the text chunks and the queries. Capabilities are captured by the nature of the sources or additional metadata that we have.

Capabilites could be more like:

1. Questions that ask about document metadata "who modified the file last"
2. Quyestions that require summarzation of a document "what are the main points of this document"
3. Questions that required timeline information "what happened in the last 3 months"
4. Questions that compare and contrast "what are the differences between these two documents"

There are all things you'll likely find as you cluster and explore the datasets.

### To be continued
