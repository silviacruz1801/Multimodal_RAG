from unstructured.partition.pdf import partition_pdf
import os
from tqdm import tqdm


class DataLoader:
    def __init__(self, files_dir, images_dir):
        print("Creando el Data Loader...\n")

        self._files_dir = files_dir
        self._images_dir = images_dir
        self._pdf_elements = self._get_elements()

    def _get_elements(self):
        files_name = os.listdir(self._files_dir)
        files = [os.path.join(self._files_dir, f) for f in files_name]

        print("Procesando los archivos para el Data Loader...\n")

        elements = []
        for _, f in tqdm(enumerate(files)):
            pdf_elements = partition_pdf(
                f,
                chunking_strategy="by_title",
                extract_images_in_pdf=True,
                max_characters=3000,
                new_after_n_chars=2800,
                combine_text_under_n_chars=2000,
                image_output_dir_path=self._images_dir
            )

            elements.extend(pdf_elements)
        
        return elements
    
    # Categorize elements by type
    def categorize_elements(self):
        """
        Categorize extracted elements from a PDF into tables and texts.
        """

        print("Categorizando los archivos para el Data Loader...\n")

        tables = []
        texts = []
        for _, pdf in tqdm(enumerate(self._pdf_elements)):
            for element in pdf:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    tables.append(str(element))
                elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    texts.append(str(element))
        return texts, tables