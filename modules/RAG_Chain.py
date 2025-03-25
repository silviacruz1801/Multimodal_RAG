import io
import re
from IPython.display import HTML, display
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from PIL import Image
import base64
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser


class RAG_Chain:
    def __init__(self, mm_llm, retriever):
        self._mm_llm = mm_llm
        self._retriever = retriever

        self._chain = self._multi_modal_rag_chain(retriever)

    def _looks_like_base64(self, sb):
        """Check if the string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


    def _is_image_data(self, b64data):
        """
        Check if the base64 data is an image by looking at the start of the data
        """
        image_signatures = {
            b"\xFF\xD8\xFF": "jpg",
            b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def _resize_base64_image(self, base64_string, size=(128, 128)):
        """
        Resize an image encoded as a Base64 string
        """
        # Decode the Base64 string
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        # Resize the image
        resized_img = img.resize(size, Image.LANCZOS)

        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)

        # Encode the resized image to Base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts
        """
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                doc = doc.page_content
            if self._looks_like_base64(doc) and self._is_image_data(doc):
                doc = self._resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
        if len(b64_images) > 0:
            return {"images": b64_images[:1], "texts": []}
        return {"images": b64_images, "texts": texts}
    
    def _img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []

        # Adding the text for analysis
        text_message = {
            "type": "text",
            "text": (
                "You are an AI scientist tasking with providing factual answers.\n"
                "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                "Use this information to provide answers related to the user question. \n"
                f"User-provided question: {data_dict['question']}\n\n"
                "Text and / or tables:\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)

        # Adding image(s) to the messages if present
        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                messages.append(image_message)

        return [HumanMessage(content=messages)]

    def _multi_modal_rag_chain(self, retriever):
        """
        Multi-modal RAG chain
        """

        print("Creando la RAG Chain...\n")

        # Multi-modal LLM
        model = ChatOllama(
            temperature=0, model=self._mm_llm, num_predict=1024
        )

        # RAG pipeline
        chain = (
            {
                "context": retriever | RunnableLambda(self._split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._img_prompt_func)
            | model
            | StrOutputParser()
        )

        return chain
    
    def get_chain(self):
        return self._chain
    
    def get_retriever(self):
        return self._retriever