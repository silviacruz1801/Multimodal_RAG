import argparse


def main():
    parser = argparse.ArgumentParser(description="Ejemplo de paquete con argumentos de consola")

    parser.add_argument(
        '--fdir',
        type=str,
        required=True,
        help="The directory of the files that will be used for the RAG"
    )
    parser.add_argument(
        '--imgdir',
        type=str,
        required=True,
        help="The directory of the images that will be extracted from the files"
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help="The query that the chatbot will answer"
    )

    args = parser.parse_args()

if __name__ == '__main__':
    main()