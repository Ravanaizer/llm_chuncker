# query_processor.py
import re
from typing import Optional

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

import config
from cache_manager import load_from_cache, save_to_cache
from prompts import get_universal_prompt


def extract_contract_from_query(query: str) -> Optional[str]:
    patterns = [
        r"договор[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"приказ[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"накладн[а-я]*\s*№\s*([\d./\-а-яА-Я]+)",
        r"№\s*([\d./\-]+/20\d{2}[а-яА-Я]?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def query_anything(index, query: str, show_debug: bool = False) -> str:
    # Проверка кэша только если включен
    if config.USE_CACHE:
        cached_answer = load_from_cache(query)
        if cached_answer:
            print("[Ответ из кэша]")
            return cached_answer

    print("Отправка запроса на сервер... (это может занять несколько минут)")

    universal_prompt = get_universal_prompt()

    # Попытка извлечь номер договора для фильтрации
    contract_number = extract_contract_from_query(query)

    if contract_number:
        print(f"Найден номер документа: {contract_number}")
        print("Фильтрация по метаданным для ускорения...")

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="contract_number", value=contract_number)]
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            filters=filters,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode="compact",
            text_qa_template=universal_prompt,
        )
    else:
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            text_qa_template=universal_prompt,
        )

    response = query_engine.query(query)

    if show_debug:
        print(f"\nНайдено {len(response.source_nodes)} источников:")
        for i, node in enumerate(response.source_nodes, 1):
            print(
                f"{i}. [{node.metadata.get('doc_type', 'N/A')}] "
                f"Договор: {node.metadata.get('contract_number', 'N/A')}, "
                f"Score: {node.score:.3f}"
            )

    answer = response.response

    # Сохранение в кэш только если включен
    if config.USE_CACHE:
        save_to_cache(query, answer)

    return answer
