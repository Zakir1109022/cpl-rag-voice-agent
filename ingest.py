from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import settings


def load_document(source: str):
    """
    Parse a document using Docling. Accepts a local file path or URL.

    Returns:
        Docling document object with structure metadata
    """
    print(f"  Parsing: {source}")
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document


def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dictionary.
    Preserves headings as a breadcrumb trail prepended to content.
    """
    headings = doc_chunk.meta.headings or []
    content = doc_chunk.text.strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings": headings,
        "content": content,
        "chunk_text": chunk_text,
    }


def ingest_documents(docs_dir: str) -> None:
    """
    Ingestion pipeline:
    1. Scan docs_dir for PDF files
    2. Parse each with Docling
    3. Chunk hierarchically
    4. Generate embeddings
    5. Index in the local Qdrant database packaged into the Docker image

    The function fails fast when source documents or chunks are missing.
    This prevents a successful Docker build from producing an image with
    an empty/missing RAG collection.
    """
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in '{docs_dir}'. "
            "Add at least one PDF to the Docker build context."
        )

    print(f"\nFound {len(pdf_files)} PDF(s) in '{docs_dir}':")
    for f in pdf_files:
        print(f"  - {f.name}")

    # Step 1: Parse and chunk all documents
    print("\n--- Step 1: Parsing & Chunking ---")
    all_chunks = []
    failed_files = []

    for pdf_path in pdf_files:
        try:
            doc = load_document(str(pdf_path))
            chunker = HierarchicalChunker(max_characters=settings.chunk_size)
            doc_chunks = [convert_chunk(c) for c in chunker.chunk(doc)]

            if not doc_chunks:
                raise RuntimeError("Document produced 0 chunks")

            print(f"  [OK] {pdf_path.name} -> {len(doc_chunks)} chunks")
            all_chunks.extend(doc_chunks)
        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            print(f"  [FAIL] {pdf_path.name}: {e}")

    if failed_files:
        failures = "; ".join(f"{name}: {error}" for name, error in failed_files)
        raise RuntimeError(f"Document ingestion failed: {failures}")

    if not all_chunks:
        raise RuntimeError(
            "No chunks were created from the source documents. "
            "Docker image cannot be built without a searchable RAG index."
        )

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 2: Generate embeddings
    print("\n--- Step 2: Generating Embeddings ---")
    embedder = SentenceTransformer(settings.embedding_model)
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
    print(f"[OK] Embedding shape: {embeddings.shape}")

    if len(embeddings) != len(all_chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(embeddings)} embeddings "
            f"for {len(all_chunks)} chunks."
        )

    # Step 3: Index in local Qdrant
    #
    # This application packages the RAG index inside the Docker image.
    # Do not switch this build to Qdrant Cloud because tools.py intentionally
    # opens the same local database at runtime.
    print("\n--- Step 3: Indexing in Qdrant ---")
    client = QdrantClient(path=settings.qdrant_path)

    print(f"  [OK] Target: local ({settings.qdrant_path})")

    dim = embedder.get_embedding_dimension()

    client.recreate_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "headings": chunk["headings"],
                "content": chunk["content"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    client.upsert(
        collection_name=settings.collection_name,
        points=points,
        wait=True,
    )

    info = client.get_collection(settings.collection_name)

    if info.points_count <= 0:
        raise RuntimeError(
            f"Qdrant collection '{settings.collection_name}' was created "
            "but contains 0 points."
        )

    print(
        f"[OK] Indexed {info.points_count} points "
        f"(dim={info.config.params.vectors.size})"
    )

    client.close()
    print("\n[DONE] Ingestion complete. The local Qdrant index is ready.")


if __name__ == "__main__":
    print("Cpl Phones — Document Ingestion Pipeline")
    print(f"  Docs folder:  {settings.docs_dir}")
    print(f"  Embedding:    {settings.embedding_model}")
    print(f"  Qdrant path:  {settings.qdrant_path}")
    print(f"  Collection:   {settings.collection_name}")

    ingest_documents(settings.docs_dir)
