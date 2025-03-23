from modules.RAG_Chain import RAG_Chain
from modules.DataRetriever import DataRetriever


def run(args):
    data_retriever = DataRetriever(args.llm, args.storage)
    retriever = data_retriever.load_retriever()

    rag_chain = RAG_Chain(args.mm_llm, retriever)
    answer = rag_chain.invoke(args.question)

    print(f"{args.mm_llm}: {answer}")