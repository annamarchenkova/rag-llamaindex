from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
    TreeSummarize,
)
from typing import List


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: TreeSummarize #BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate
    postprocessors: List[BaseNodePostprocessor]

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        for postprocessor in self.postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes)
            # print(nodes)
            # print(len(nodes))

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response), nodes
