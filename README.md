# Multimodal_RAG

Implementación de un sistema RAG con LangChain y Ollama en donde se pretende implementar una base de datos vectorial con Chroma para realizarle queries a un LLM multimodal. Para ello, se procesan los pdfs con la librería Unstructured, se realiza un resumen de texto, imágenes y tablas con otros LLMs, se crea la base de datos vectorial con esta información y, finalmente, se realizan las queries correspondientes.
