import argparse
from commands import create, query


def main():
    parser = argparse.ArgumentParser(
        description="CLI para usar un chatbot con RAG multimodal implementado"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Comandos disponibles")

    parser_create = subparsers.add_parser("create", help="Crea y guarda en disco el retriever")
    parser_create.add_argument(
        "--fdir",
        type=str,
        default="./data",
        help="Directorio donde guardar los archivos usados para el RAG"
    )
    parser_create.add_argument(
        "--imgdir",
        type=str,
        default="./figures",
        help="Directorio donde guardar las imagenes"
    )
    parser_create.add_argument(
        "--storage",
        type=str,
        default="./storage",
        help="Directorio donde guardar el retriever"
    )
    parser_create.add_argument(
        "--llm",
        type=str,
        default="llama3.2",
        help="LLM de Ollama que se usará como parte del proceso de RAG"
    )
    parser_create.add_argument(
        "--mmllm",
        type=str,
        default="llava",
        help="LLM multimodal que se usará como parte del proceso de RAG"
    )
    parser_create.set_defaults(func=create.run)

    parser_query = subparsers.add_parser("query", help="Realiza una query al chatbot")
    parser_query.add_argument(
        "--storage",
        type=str,
        default="./storage",
        help="Directorio de donde cargar el retriever"
    )
    parser_query.add_argument(
        "--llm",
        type=str,
        default="llama3.2",
        help="LLM de Ollama que se usará como parte del proceso de RAG"
    )
    parser_query.add_argument(
        "--mmllm",
        type=str,
        default="llava",
        help="LLM multimodal que se usará como chatbot final"
    )
    parser_query.set_defaults(func=query.run)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()