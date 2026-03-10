<div align="center">

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://mdn.alipayobjects.com/huamei_ytl0i7/afts/img/A*pKqtRILxGioAAAAAQLAAAAgAejCYAQ/original" width="420">
  <source media="(prefers-color-scheme: light)" srcset="https://mdn.alipayobjects.com/huamei_ytl0i7/afts/img/A*6BO4Q6D78GQAAAAAQFAAAAgAejCYAQ/original" width="420">
  <img alt="示意图" src="light-mode.png">
</picture> -->

### **🔷 The AI-Native Search Database**

**Unifies vector, text, structured and semi-structured data in a single engine, enabling hybrid search and in-database AI workflows.**

**在一个数据库中融合向量、文本、结构化与半结构化数据能力，并通过内置 AI Functions 支持多模混合搜索与智能推理。**

</div>

---
<div align="center">
<p>
    <a href="https://oceanbase.ai">
        <img alt="Documentation" height="20" src="https://img.shields.io/badge/OceanBase.ai-4285F4?style=for-the-badge&logo=read-the-docs&logoColor=white" />
    </a>
    <a href="https://www.linkedin.com/company/oceanbase" target="_blank">
        <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" alt="follow on LinkedIn">
    </a>
    <a href="https://www.youtube.com/@OceanBaseDB">
        <img alt="Static Badge" src="https://img.shields.io/badge/YouTube-red?logo=youtube">
    </a>
    <a href="https://deepwiki.com/oceanbase/seekdb">
        <img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg" />
    </a>
    <!-- <a href="https://discord.gg/74cF8vbNEs">
        <img alt="Join Discord" src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&style=flat-square" />
    </a> -->
    <!-- <a href="https://pepy.tech/projects/pylibseekdb">
        <img height="20" alt="Downloads" src="https://static.pepy.tech/badge/pylibseekdb" />
    </a> -->
    <a href="https://github.com/oceanbase/seekdb/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" />
    </a>
</p>
</div>

<div align="center">



---

</div>

## 🚀 What is OceanBase Geekdb?

**OceanBase Geekdb** is a distribution built on **OceanBase seekdb** that supports more leading "cool" features on top of SeekDB's foundation.

**GeekDB** 是一个基于 **OceanBase seekdb** 制作的发行版，在 **seekdb** 的基础之上支持了更多领先的「酷」特性。

---

## 🔥 Why OceanBase Geekdb?

| **Feature**              | **Geekdb** | **seekdb** | **OceanBase** | **Chroma** | **Milvus** | **MySQL&nbsp;9.0**           | **PostgreSQL<br/>+pgvector** | **DuckDB** | **Elasticsearch**                   |
| ------------------------ |:--------------------:|:--------------------:|:-------------:|:----------:|:----------:|:-----------------------:|:----------------------------:|:----------:|:-----------------------------------:|
| **Mac GPU Acceleration**    |✅ | ❌ | ❌ |❌ | ❌ | ❌ | ❌| ❌ | ❌ |
| **Embedded**    | ✅                    | ✅                    | ❌             | ✅          | ✅          | ❌<sup>[1]</sup> | ❌                            | ✅          | ❌                                   |
| **Single-Node** | ✅                    | ✅                    | ✅             | ✅          | ✅          | ✅                       | ✅                            | ✅          | ✅                                   |
| **Distributed** | ❌                    | ❌                    | ✅             | ❌          | ✅          | ❌                       | ❌                            | ❌          | ✅                                   |
| **MySQL&nbsp;Compatible**   | ✅                    | ✅                    | ✅             | ❌          | ❌          | ✅                       | ❌                            | ✅          | ❌                                   |
| **Vector&nbsp;Search**     | ✅                    | ✅                    | ✅             | ✅          | ✅          | ❌                       | ✅                            | ✅          | ✅                                   |
| **Full-Text&nbsp;Search**    | ✅                    | ✅                    | ✅             | ✅          | ⚠️         | ✅                       | ✅                            | ✅          | ✅                                   |
| **Hybrid&nbsp;Search** | ✅                    | ✅                    | ✅             | ✅          | ✅          | ❌                       | ⚠️                           | ❌          | ✅                                   |
| **OLTP**                 | ✅                    | ✅                    | ✅             | ❌          | ❌          | ✅                       | ✅                            | ❌          | ❌                                   |
| **OLAP**                 | ✅                    | ✅                    | ✅             | ❌          | ❌          | ❌                       | ✅                            | ✅          | ⚠️                                  |
| **License**  | Apache 2.0           | Apache 2.0           | MulanPubL 2.0 | Apache 2.0 | Apache 2.0 | GPL 2.0                 | PostgreSQL License           | MIT        | AGPLv3<br/>+SSPLv1<br/>+Elastic 2.0 |
> [1] Embedded capability is removed in MySQL 8.0
> - ✅ Supported
> - ❌ Not Supported
> - ⚠️ Limited

## ✨ Key Features

### Hybrid search + Multi model
1. **Hybrid Search:** Combine vector search, full-text search and relational query in a single statement.
2. **Multi-Model:** Support relational, vector, text, JSON and GIS in a single engine.


### AI inside + SQL inside
1. **AI Inside:** Run embedding, reranking, LLM inference and prompt management inside the database, supporting a complete document-in/data-out RAG workflow.
2. **SQL Inside:** 	Powered by the proven OceanBase engine, delivering real-time writes and queries with full ACID compliance, and seamless MySQL ecosystem compatibility.

### GPU Empowered Vector Index Building/GPU加速向量索引构建
1. **GPU IVF Index Building:** Build 1 Million rows IVF index on Mac M3 Pro with Metal in 40 seconds. 40秒在Mac M3 Pro上使用Metal构建百万行IVF索引

---

## 🎬 Quick Start

### Build from Source
#### Geekdb supports MacOS with M series Chips and Metal(GPU) available only.

Before building, please install the required toolchain and dependencies for your operating system. See [Install Toolchain](docs/developer-guide/en/toolchain.md) for detailed instructions.

```bash
# Clone the repository
brew install git cmake pkg-config openssl@3 ncurses googletest
brew install zstd utf8proc thrift re2 brotli
git clone git@github.com:JasonZhang10086/Geekdb.git
cd Geekdb
bash build.sh release --init --make
mkdir ~/seekdb
mkdir ~/seekdb/bin
cp build_release/src/observer/seekdb ~/seekdb/bin
cd ~/seekdb
./bin/seekdb
mysql -uroot -h127.0.0.1 -P2881
```

In this example, the working director is $HOME/seekdb, please use a fresh director for testing, Please see the [Developer Guide](docs/developer-guide/en/README.md) for detailed instructions.

### 🎯 AI Search Example

Build a semantic search system in 5 minutes:

<details>
<summary><b>🗄️ 🐍 Python SDK</b></summary>

```bash
# install sdk first
pip install -U pyseekdb
```

```python
"""
this example demonstrates the most common operations with embedding functions:
1. Create a client connection
2. Create a collection with embedding function
3. Add data using documents (embeddings auto-generated)
4. Query using query texts (embeddings auto-generated)
5. Print query results

This is a minimal example to get you started quickly with embedding functions.
"""

import pyseekdb
from pyseekdb import DefaultEmbeddingFunction

# ==================== Step 1: Create Client Connection ====================
# You can use embedded mode, server mode, or OceanBase mode
# For this example, we'll use server mode (you can change to embedded or OceanBase)

# Embedded mode (local SeekDB)
client = pyseekdb.Client(
    path="./seekdb.db",
    database="test"
)
# Alternative: Server mode (connecting to remote SeekDB server)
# client = pyseekdb.Client(
#     host="127.0.0.1",
#     port=2881,
#     database="test",
#     user="root",
#     password=""
# )

# Alternative: Remote server mode (OceanBase Server)
# client = pyseekdb.Client(
#     host="127.0.0.1",
#     port=2881,
#     tenant="test",  # OceanBase default tenant
#     database="test",
#     user="root",
#     password=""
# )

# ==================== Step 2: Create a Collection with Embedding Function ====================
# A collection is like a table that stores documents with vector embeddings
collection_name = "my_simple_collection"

# Create collection with default embedding function
# The embedding function will automatically convert documents to embeddings
collection = client.create_collection(
    name=collection_name,
    #embedding_function=DefaultEmbeddingFunction()  # Uses default model (384 dimensions)
)

print(f"Created collection '{collection_name}' with dimension: {collection.dimension}")
print(f"Embedding function: {collection.embedding_function}")

# ==================== Step 3: Add Data to Collection ====================
# With embedding function, you can add documents directly without providing embeddings
# The embedding function will automatically generate embeddings from documents

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Vector databases enable semantic search",
    "Neural networks are inspired by the human brain",
    "Natural language processing helps computers understand text"
]

ids = ["id1", "id2", "id3", "id4", "id5"]

# Add data with documents only - embeddings will be auto-generated by embedding function
collection.add(
    ids=ids,
    documents=documents,  # embeddings will be automatically generated
    metadatas=[
        {"category": "AI", "index": 0},
        {"category": "Programming", "index": 1},
        {"category": "Database", "index": 2},
        {"category": "AI", "index": 3},
        {"category": "NLP", "index": 4}
    ]
)

print(f"\nAdded {len(documents)} documents to collection")
print("Note: Embeddings were automatically generated from documents using the embedding function")

# ==================== Step 4: Query the Collection ====================
# With embedding function, you can query using text directly
# The embedding function will automatically convert query text to query vector

# Query using text - query vector will be auto-generated by embedding function
query_text = "artificial intelligence and machine learning"

results = collection.query(
    query_texts=query_text,  # Query text - will be embedded automatically
    n_results=3  # Return top 3 most similar documents
)

print(f"\nQuery: '{query_text}'")
print(f"Query results: {len(results['ids'][0])} items found")

# ==================== Step 5: Print Query Results ====================
for i in range(len(results['ids'][0])):
    print(f"\nResult {i+1}:")
    print(f"  ID: {results['ids'][0][i]}")
    print(f"  Distance: {results['distances'][0][i]:.4f}")
    if results.get('documents'):
        print(f"  Document: {results['documents'][0][i]}")
    if results.get('metadatas'):
        print(f"  Metadata: {results['metadatas'][0][i]}")

# ==================== Step 6: Cleanup ====================
# Delete the collection
client.delete_collection(collection_name)
print(f"\nDeleted collection '{collection_name}'")

```
Please refer to the [User Guide](https://github.com/oceanbase/pyseekdb) for more details.
</details>

<details>
<summary><b>🗄️ SQL</b></summary>

```sql
-- Create table with vector column
CREATE TABLE articles (
            id INT PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding VECTOR(384),
            FULLTEXT INDEX idx_fts(content) WITH PARSER ik,
            VECTOR INDEX idx_vec (embedding) WITH(DISTANCE=l2, TYPE=hnsw, LIB=vsag)
        ) ORGANIZATION = HEAP;

-- Insert documents with embeddings
-- Note: Embeddings should be pre-computed using your embedding model
INSERT INTO articles (id, title, content, embedding)
VALUES
    (1, 'AI and Machine Learning', 'Artificial intelligence is transforming...', '[0.1, 0.2, ...]'),
    (2, 'Database Systems', 'Modern databases provide high performance...', '[0.3, 0.4, ...]'),
    (3, 'Vector Search', 'Vector databases enable semantic search...', '[0.5, 0.6, ...]');

-- Example: Hybrid search combining vector and full-text
-- Replace '[query_embedding]' with your actual query embedding vector
SELECT
    title,
    content,
    l2_distance(embedding, '[query_embedding]') AS vector_distance,
    MATCH(content) AGAINST('your keywords' IN NATURAL LANGUAGE MODE) AS text_score
FROM articles
WHERE MATCH(content) AGAINST('your keywords' IN NATURAL LANGUAGE MODE)
ORDER BY vector_distance APPROXIMATE
LIMIT 10;
```
We suggest developers use sqlalchemy to access data by SQL for python developers.
</details>


## 📚 Use Cases

<details>
<summary><b> 📖 RAG & Knowledge Retrieval</b></summary>

Large language models are limited by their training data. RAG introduces timely and trusted external knowledge to improve answer quality and reduce hallucination. seekdb enhances search accuracy through vector search, full-text search, hybrid search, built-in AI functions, and efficient indexing, while multi-level access control safeguards data privacy across heterogeneous knowledge sources.
1. Enterprise QA
2. Customer support
3. Industry insights
4. Personal knowledge

</details>

<details>
<summary><b> 🔍 Semantic Search Engine</b></summary>

Traditional keyword search struggles to capture intent. Semantic search leverages embeddings and vector search to understand meaning and connect text, images, and other modalities. seekdb's hybrid search and multi-model querying deliver more precise, context-aware results across complex search scenarios.
1. Product search
2. Text-to-image
3. Image-to-product

</details>

<details>
<summary><b> 🎯 Agentic AI Applications</b></summary>

Agentic AI requires memory, planning, perception, and reasoning. seekdb provides a unified foundation for agents through metadata management, vector/text/mixed queries, multimodal data processing, RAG, built-in AI functions and inference, and robust privacy controls—enabling scalable, production-grade agent systems.
1. Personal assistants
2. Enterprise automation
3. Vertical agents
4. Agent platforms

</details>

<details>
<summary><b> 💻 AI-Assisted Coding & Development</b></summary>

AI-powered coding combines natural-language understanding and code semantic analysis to enable generation, completion, debugging, testing, and refactoring. seekdb enhances code intelligence with semantic search, multi-model storage for code and documents, isolated multi-project management, and time-travel queries—supporting both local and cloud IDE environments.
1. IDE plugins
2. Design-to-web
3. Local IDEs
4. Web IDEs

</details>

<details>
<summary><b> ⬆️ Enterprise Application Intelligence</b></summary>

AI transforms enterprise systems from passive tools into proactive collaborators. seekdb provides a unified AI-ready storage layer, fully compatible with MySQL syntax and views, and accelerates mixed workloads with parallel execution and hybrid row-column storage. Legacy applications gain intelligent capabilities with minimal migration across office, workflow, and business analytics scenarios.
1. Document intelligence
2. Business insights
3. Finance systems

</details>


<details>
<summary><b> 📱 On-Device & Edge AI Applications</b></summary>

Edge devices—from mobile to vehicle and industrial terminals—operate with constrained compute and storage. seekdb's lightweight architecture supports embedded and micro-server modes, delivering full SQL, JSON, and hybrid search under low resource usage. It integrates seamlessly with OceanBase cloud services to enable unified edge-to-cloud intelligent systems.
1. Personal assistants
2. In-vehicle systems
3. AI education
4. Companion robots
5. Healthcare devices

</details>



---


## 📄 License

OceanBase seekdb is licensed under the [Apache License, Version 2.0](LICENSE).


