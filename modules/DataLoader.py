from unstructured.partition.pdf import partition_pdf
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from PIL import ImageFile
import concurrent.futures


class DataLoader:
    def __init__(self, files_dir, images_dir):
        self._files_dir = files_dir
        self._images_dir = images_dir
        self._pdf_elements = self._get_elements()

    def _process_pdf(self, file_path):
        pdf_elements = partition_pdf(
            file_path,
            chunking_strategy="by_title",
            extract_images_in_pdf=True,
            max_characters=3000,
            new_after_n_chars=2800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=self._images_dir
        )

        return pdf_elements

    def _get_elements(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        files_name = os.listdir(self._files_dir)
        files = [os.path.join(self._files_dir, f) for f in files_name]

        with Progress(
            TextColumn("[bold blue]{task.description}"),  
            BarColumn(),                                  
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeRemainingColumn()                         
        ) as progress:
            # Creamos una tarea con un total definido (por ejemplo, 100 unidades)
            task = progress.add_task("Procesando los archivos para el Data Loader...", 
                                     total=len(files))
            
            elements = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(self._process_pdf, pdf): pdf for pdf in files}

                for future in concurrent.futures.as_completed(futures):
                    el = future.result()
                    elements.append(el)
                    progress.advance(task)
        
        return elements
    
    # Categorize elements by type
    def categorize_elements(self):
        """
        Categorize extracted elements from a PDF into tables and texts.
        """

        print("Categorizando los archivos para el Data Loader...\n")
        with Progress(
            TextColumn("[bold blue]{task.description}"),  
            BarColumn(),                                  
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeRemainingColumn()                         
        ) as progress:
            # Creamos una tarea con un total definido (por ejemplo, 100 unidades)
            task = progress.add_task("Categorizando los archivos para el Data Loader...", 
                                     total=len(self._pdf_elements))
            tables = []
            texts = []
            for pdf in self._pdf_elements:
                for element in pdf:
                    if "unstructured.documents.elements.Table" in str(type(element)):
                        tables.append(str(element))
                    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                        texts.append(str(element))
                progress.update(task, advance=1)

        return texts, tables