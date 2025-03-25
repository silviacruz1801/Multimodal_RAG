from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama.llms import OllamaLLM

import base64
import os
from langchain_core.messages import HumanMessage
from tqdm import tqdm


class DataSummarizer:
    def __init__(self, data_loader, texts, tables, llm, mm_llm):
        self._data_loader = data_loader
        self._texts = texts
        self._tables = tables
        self._llm = llm
        self._mm_llm = mm_llm

    # Generate summaries of text elements
    def _generate_text_summaries(self, summarize_texts=False):
        """
        Summarize text elements
        summarize_texts: Bool to summarize texts
        """

        print("Resumiendo los archivos de texto...\n")

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well-optimized for retrieval. Table \
        or text: {element} """
        prompt = PromptTemplate.from_template(prompt_text)
        empty_response = RunnableLambda(
            lambda x: AIMessage(content="Error processing document")
        )
        # Text summary chain
        model = OllamaLLM(temperature=0, model=self._llm, num_predict=1024
                        ).with_fallbacks([empty_response])
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if self._texts and summarize_texts:
            text_summaries = summarize_chain.batch(self._texts, {"max_concurrency": 1})
        elif self._texts:
            text_summaries = self._texts

        # Apply to tables if tables are provided
        if self._tables:
            table_summaries = summarize_chain.batch(self._tables, {"max_concurrency": 1})

        return text_summaries, table_summaries
    
    def _encode_image(self, image_path):
        """Getting the base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def _image_summarize(self, img_base64, prompt):
        """Make image summary"""
        model = ChatOllama(model=self._mm_llm, temperature=0, num_predict=1024)

        msg = model(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content

    def _generate_img_summaries(self, path):
        """
        Generate summaries and base64 encoded strings for images
        path: Path to list of .jpg files extracted by Unstructured
        """

        print("Resumiendo las imágenes...\n")

        # Store base64 encoded images
        img_base64_list = []

        # Store image summaries
        image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        # Apply to images
        for _, img_file in tqdm(enumerate(sorted(os.listdir(path)))):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = self._encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(self._image_summarize(base64_image, prompt))

        return img_base64_list, image_summaries
    
    def generate_summaries(self, img_path):
        text_summaries, table_summaries = self._generate_text_summaries(self._texts, self._tables)
        img_base64_list, image_summaries = self._generate_img_summaries(img_path)

        return text_summaries, table_summaries, img_base64_list, image_summaries