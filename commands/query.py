from modules.RAG_Chain import RAG_Chain


def run(args):
    rag_chain = RAG_Chain(args.mm_llm, data_retriever.get_retriever())