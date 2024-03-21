# Duck DB test
from lancedb import connect
import duckdb

db = connect("./db")

chunk_table = db.open_table("pg")
doc_table = db.open_table("document")
## Test that we can do simple select queries
chunks = chunk_table.to_lance()
docs = doc_table.to_lance()

queries = [
    "SELECT * FROM docs",
    "SELECT doc_id, count(chunk_id) as count FROM chunks GROUP BY doc_id",
    "SELECT * FROM docs WHERE metadata->>'date' LIKE '2021%';",
    "SELECT * FROM chunks INNER JOIN docs on chunks.doc_id = docs.id LIMIT 10",
    "SELECT * FROM chunks INNER JOIN docs on chunks.doc_id = docs.id WHERE doc_id = '897b17269004b437f58b5bf1f883e2dc' LIMIT 10 ",
]

for query in queries:
    print(f"EXECUTING >> {query}")
    print(duckdb.query(query))
