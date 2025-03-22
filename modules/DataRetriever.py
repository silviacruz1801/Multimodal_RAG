from langchain.schema.document import Document
import uuid
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


class DataRetriever:
    def __init__(self, text_summaries, texts, table_summaries, 
                 tables, image_summaries, images, llm, persist_dir):
        self._text_summaries = text_summaries
        self._texts = texts
        self._table_summaries = table_summaries
        self._tables = tables
        self._image_summaries = image_summaries
        self._images = images
        self._llm = llm
        self._persist_dir = persist_dir

        self._retriever = self._create_multi_vector_retriever()


    def _create_multi_vector_retriever(self):
        """
        Create retriever that indexes summaries, but returns raw images or texts
        """

        embeddings = OllamaEmbeddings(model=self._llm)

        # The vectorstore to use to index the summaries
        vectorstore = Chroma(
            collection_name="mm_rag_mistral",
            embedding_function=embeddings,
        )

        # Initialize the storage layer
        store = InMemoryStore()
        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            persist_directory=self._persist_dir,
            docstore=store,
            id_key=id_key,
        )

        # Helper function to add documents to the vectorstore and docstore
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

        # Add texts, tables, and images
        # Check that text_summaries is not empty before adding
        if self._text_summaries:
            add_documents(retriever, self._text_summaries, self._texts)
        # Check that table_summaries is not empty before adding
        if self._table_summaries:
            add_documents(retriever, self._table_summaries, self._tables)
        # Check that image_summaries is not empty before adding
        if self._image_summaries:
            add_documents(retriever, self._image_summaries, self._images)

        vectorstore.persist()
        return retriever
    
    def get_retriever(self):
        return self._retriever