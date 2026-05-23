from __future__ import annotations

import argparse

from app.controllers.build_controller import BuildController
from app.core.dependencies import get_build_service


def refresh_all_uploaded_documents() -> dict:
    controller = BuildController(get_build_service())
    return controller.build_files(background_tasks=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh embeddings/vector index for uploaded documents."
    )
    parser.parse_args()
    result = refresh_all_uploaded_documents()
    print(result)


if __name__ == "__main__":
    main()
