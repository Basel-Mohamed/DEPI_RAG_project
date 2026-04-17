rag-agent/
│
├── app/
│
│   ├── api/                     # FastAPI routes
│   │   ├── build_routes.py
│   │   ├── inference_routes.py
│   │   ├── monitoring_routes.py
│   │   └── router.py
│
│   ├── core/                    # config & shared logic
│   │   ├── config.py
│   │   ├── settings.py
│   │   └── logger.py
│
│   ├── domain/                  # interfaces & entities
│   │   ├── models/
│   │   │   ├── document.py
│   │   │   ├── query.py
│   │   │   └── response.py
│   │   │
│   │   └── interfaces/
│   │       ├── embedding_interface.py
│   │       ├── llm_interface.py
│   │       ├── vector_db_interface.py
│   │       └── reranker_interface.py
│
│   ├── services/                # business logic
│   │
│   │   ├── preprocessing/
│   │   │   ├── document_loader.py
│   │   │   ├── pdf_parser.py
│   │   │   └── chunking_service.py
│   │
│   │   ├── embedding/
│   │   │   └── embedding_service.py
│   │
│   │   ├── vector_db/
│   │   │   └── vector_store_service.py
│   │
│   │   ├── reranking/
│   │   │   └── reranker_service.py
│   │
│   │   ├── rag/
│   │   │   ├── rag_build_service.py
│   │   │   ├── rag_inference_service.py
│   │   │   └── context_builder.py
│   │
│   │   ├── llm/
│   │   │   └── llm_service.py
│   │
│   │   └── monitoring/
│   │       └── monitoring_service.py
│
│   ├── infrastructure/          # external integrations
│   │
│   │   ├── embeddings/
│   │   │   └── cohere_embeddings.py
│   │
│   │   ├── llms/
│   │   │   └── cohere_llm.py
│   │
│   │   ├── vector_db/
│   │   │   └── qdrant_client.py
│   │
│   │   ├── rerankers/
│   │   │   └── cohere_reranker.py
│   │
│   │   └── monitoring/
│   │       └── grafana_client.py
│
│   └── pipelines/               # orchestrating flows
│       ├── rag_build_pipeline.py
│       └── rag_inference_pipeline.py
│
│
├── tests/
│
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_embedding.py
│   │   └── test_reranking.py
│
│   ├── integration/
│   │   └── test_rag_pipeline.py
│
│   └── e2e/
│       └── test_full_system.py
│
│
├── scripts/
│   ├── build_index.py
│   └── run_inference.py
│
│
├── deployment/
│   ├── dockerfile
│   ├── docker-compose.yml
│   └── aws/
│       ├── ecs.tf
│       └── infrastructure.tf
│
│
├── frontend/                   # optional
│   ├── src/
│   └── package.json
│
│
├── requirements.txt
├── main.py
└── README.md