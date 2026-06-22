# index_manager.py
from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from metadata_extractor import extract_metadata


def get_or_create_index():
    chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )

    if chroma_collection.count() > 0:
        print("Загрузка существующего индекса из ChromaDB...")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return VectorStoreIndex.from_vector_store(vector_store)

    print("Индекс не найден. Начинаем создание...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context)

    md_path = Path(config.MD_DIR)
    if not md_path.exists():
        raise FileNotFoundError(f"Папка '{config.MD_DIR}' не найдена!")

    count = 0
    supported_extensions = {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf"}

    for file_path in md_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            raw_docs = reader.load_data()

            for doc in raw_docs:
                metadata = extract_metadata(doc.text, doc.metadata.get("file_name", ""))

                enriched_text = f"""
doc_type: {metadata.get("doc_type", "Неизвестно")}
contract_number: {metadata.get("contract_number", "Не указан")}
contract_date: {metadata.get("contract_date", "Не указана")}
organization: {metadata.get("organization", "Не указана")}
amount: {metadata.get("amount", "Не указана")}
{doc.text}
"""
                enriched_doc = Document(
                    text=enriched_text, metadata={**doc.metadata, **metadata}
                )

                index.insert(enriched_doc)
                count += 1

            if count % 10 == 0:
                print(f"Обработано {count} файлов...")

    print(f"Индекс успешно создан и сохранен в {config.PERSIST_DIR}")
    return index


def add_document(file_path: str, index) -> bool:
    """Add a single document to the existing index."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Ошибка: файл {file_path} не найден")
        return False

    if file_path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf"}:
        print(f"Ошибка: неподдерживаемый формат {file_path.suffix}")
        return False

    try:
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        raw_docs = reader.load_data()

        for doc in raw_docs:
            metadata = extract_metadata(doc.text, doc.metadata.get("file_name", ""))

            enriched_text = f"""
ТИП: {metadata.get("doc_type", "Неизвестно")}
НОМЕР: {metadata.get("contract_number", "Не указан")}
ДАТА: {metadata.get("contract_date", "Не указана")}
ОРГАНИЗАЦИЯ: {metadata.get("organization", "Не указана")}
СУММА: {metadata.get("amount", "Не указана")}
{doc.text}
"""
            enriched_doc = Document(
                text=enriched_text, metadata={**doc.metadata, **metadata}
            )
            index.insert(enriched_doc)

        # ChromaDB PersistentClient автоматически сохраняет изменения
        print(f"Документ {file_path.name} успешно добавлен в индекс")
        return True
    except Exception as e:
        print(f"Ошибка при добавлении документа: {e}")
        return False


def remove_document_by_filename(filename: str, index) -> bool:
    """Remove all document chunks with the given filename from the index using ChromaDB filter."""
    try:
        # Get ChromaDB collection directly from vector store
        vector_store = index._vector_store
        chroma_collection = vector_store._collection

        # Delete all documents with matching filename using ChromaDB filter
        # This is much faster than iterating through all documents
        result = chroma_collection.delete(where={"file_name": filename})

        if result and hasattr(result, "deleted_count"):
            print(f"Удалено {result.deleted_count} фрагментов документа '{filename}'")
        else:
            print(f"Документ '{filename}' удален из индекса (если существовал)")

        return True
    except Exception as e:
        print(f"Ошибка при удалении документа: {e}")
        return False


def list_documents_in_index(index) -> list:
    """List all unique filenames currently in the index."""
    try:
        vector_store = index._vector_store
        chroma_collection = vector_store._collection

        # Get all unique filenames from metadata
        all_docs = chroma_collection.get(include=["metadatas"])

        filenames = set()
        for metadata in all_docs.get("metadatas", []):
            if metadata and "file_name" in metadata:
                filenames.add(metadata["file_name"])

        return sorted(list(filenames))
    except Exception as e:
        print(f"Ошибка при получении списка документов: {e}")
        return []


def rebuild_index():
    """Delete existing index and rebuild from scratch."""
    import shutil

    if os.path.exists(config.PERSIST_DIR):
        print(f"Удаление старого индекса из {config.PERSIST_DIR}...")
        shutil.rmtree(config.PERSIST_DIR)

    print("Пересоздание индекса...")
    return get_or_create_index()
