from modules.DataLoader import DataLoader
from modules.DataSummarizer import DataSummarizer
from modules.DataRetriever import DataRetriever


def run(args):
    data_loader = DataLoader(args.fdir, args.imgdir)

    texts, tables = data_loader.categorize_elements()
    data_summarizer = DataSummarizer(data_loader, texts, tables, args.llm, args.mm_llm)
    text_summaries, table_summaries, img_base64_list, image_summaries = data_summarizer.generate_summaries(args.imgdir)

    data_retriever = DataRetriever(args.llm, args.storage)
    data_retriever.create_retriever(text_summaries, texts, table_summaries, 
                                                 tables, image_summaries, img_base64_list)

    

