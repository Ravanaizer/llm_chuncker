# llm_setup.py
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

import config


def initialize_llm():
    llm = OpenAILike(
        model="openai/gpt-oss-120b",
        api_base=config.API_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        timeout=1200.0,
    )
    return llm


def initialize_embed_model():
    embed_model = OpenAIEmbedding(
        api_base=config.API_BASE_URL,
        api_key="dummy_key",
        model_name="text-embedding-nomic-embed-text-v1.5",
    )
    return embed_model


def setup_settings():
    llm = initialize_llm()
    embed_model = initialize_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024, chunk_overlap=200
    )

    return llm, embed_model
