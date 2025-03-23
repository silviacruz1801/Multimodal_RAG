from langchain.schema.document import Document
import uuid
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import pickle
import os


class DataRetriever:
    def __init__(self, llm, persist_dir):
        self._llm = llm
        self._persist_dir = persist_dir


    def create_retriever(self, text_summaries, texts, table_summaries, 
                 tables, image_summaries, images):
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
        if text_summaries:
            add_documents(retriever, text_summaries, texts)
        # Check that table_summaries is not empty before adding
        if table_summaries:
            add_documents(retriever, table_summaries, tables)
        # Check that image_summaries is not empty before adding
        if image_summaries:
            add_documents(retriever, image_summaries, images)

        with open(os.path.join(self._persist_dir, "docstore.pkl"), "wb") as f:
            pickle.dump(store, f)

        vectorstore.persist()
        return retriever
        
    def load_retriever(self):
        if os.path.exists(self._persist_dir) and os.listdir(self._persist_dir):
            raise Exception("El directorio del retriever no existe o está vacío")
        
        embeddings = OllamaEmbeddings(model=self._llm)

        vectorstore = Chroma(
            persist_directory=self._persist_dir,
            embedding_function=embeddings,
        )

        with open(os.path.join(self._persist_dir, "docstore.pkl"), "rb") as f:
            store = pickle.load(f)

        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            persist_directory=self._persist_dir,
            docstore=store,
            id_key=id_key,
        )

        return retriever