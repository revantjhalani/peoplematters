from llama_index.llms.openai import OpenAI
from llama_index.core.indices.struct_store import JSONQueryEngine

llm = OpenAI(model="gpt-4")

nl_query_engine = JSONQueryEngine(
    json_value=json_value,
    json_schema=json_schema,
    llm=llm,
)
raw_query_engine = JSONQueryEngine(
    json_value=json_value,
    json_schema=json_schema,
    llm=llm,
    synthesize_response=False,
)


nl_response = nl_query_engine.query(
    "What comments has Jerry been writing?",
)
raw_response = raw_query_engine.query(
    "What comments has Jerry been writing?",
)