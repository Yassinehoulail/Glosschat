

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="Reconstruire l'index Chroma")
    index_parser.add_argument("--batch-size", type=int, default=100)
    ask_parser = commands.add_parser("ask", help="Poser une question au RAG")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Search every eligible catalogue record before ranking the answer context.",
    )
    query_parser = commands.add_parser(
        "query", help="Afficher la StructuredQuery produite par LangChain"
    )
    query_parser.add_argument("question")
    args = parser.parse_args()

    if getattr(args, "exhaustive", False):
        # Settings are created during the imports below.
        os.environ["RAG_EXHAUSTIVE_SEARCH"] = "true"

    if args.command == "index":
        from vector_store import rebuild_vector_store

        print(f"{rebuild_vector_store(args.batch_size)} KPI indexés.")
    elif args.command == "query":
        from retriever import build_structured_query

        print(build_structured_query(args.question))
    else:
        from rag import ask

        print(ask(args.question))


if __name__ == "__main__":
    main()
