---
id: 2
title: "Plan - Book RAG Ingestion System"
stage: "plan"
date: "2025-12-16"
model: "Claude Sonnet 4.5"
feature: "001-book-rag-ingestion"
branch: "001-book-rag-ingestion"
user: "user"
command: "sp.plan"
labels: ["specification", "planning", "architecture", "book-ingestion"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "specs/001-book-rag-ingestion/plan.md"
  - "specs/001-book-rag-ingestion/spec.md"
  - "specs/001-book-rag-ingestion/data-model.md"
  - "specs/001-book-rag-ingestion/research.md"
tests: []
---

# Plan - Book RAG Ingestion System

## Prompt
Create an implementation plan for the Book Content Ingestion, Embedding Generation, and Vector Database Storage feature based on the specification.

## Response
# Implementation Plan: Book Content Ingestion System

**Branch**: `001-book-rag-ingestion` | **Date**: 2025-12-15 | **Spec**: [link](/mnt/d/Hackathon/book/specs/001-book-rag-ingestion/spec.md)
**Input**: Feature specification from `/specs/001-book-rag-ingestion/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a single-file Python system (`main.py`) to extract content from Docusaurus-based book websites (specifically https://sanilahmed.github.io/hackathon-ai-book/), chunk and embed the content using Cohere models, and store the embeddings in Qdrant Cloud for RAG applications. The system will be managed with the uv package manager.

## Technical Context

**Language/Version**: Python 3.10+ (as specified in feature constraints)
**Primary Dependencies**: uv package manager, requests, beautifulsoup4, cohere, qdrant-client, python-dotenv
**Storage**: Qdrant Cloud vector database
**Testing**: pytest with integration tests for API connections
**Target Platform**: Linux server environment
**Project Type**: Backend service (single file implementation)
**Performance Goals**: Process and embed content within reasonable timeframes respecting API rate limits
**Constraints**: Must handle API rate limits, memory usage for large books, and network reliability
**Scale/Scope**: Process entire book site with multiple pages, store embeddings with metadata

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Project uses Python 3.10+ as specified in constitution
- [x] Dependencies align with active technologies in constitution (Python, Cohere, Qdrant)
- [x] Implementation approach follows specified single-file requirement
- [x] No architectural violations detected

## Implementation Approach

### Core Functions to Implement
1. `get_all_urls(base_url)` - Discover and validate all URLs from the Docusaurus book
2. `extract_text_from_url(url)` - Extract and clean text content from each URL
3. `chunk_text(text, chunk_size=1000)` - Split content into fixed-size segments
4. `embed(texts)` - Generate embeddings using Cohere API
5. `create_collection(client, collection_name="rag_embedding")` - Create Qdrant collection
6. `save_chunk_to_qdrant(client, collection_name, chunk, embedding, metadata)` - Store embeddings
7. `main()` - Execute complete pipeline

### Architecture Decision
- Single-file implementation in main.py as explicitly requested
- All functions in one file for simplicity and easy deployment
- Proper error handling and logging throughout
- Environment variable configuration for API keys
- Target Site: https://sanilahmed.github.io/hackathon-ai-book/
- SiteMap Url: https://sanilahmed.github.io/hackathon-ai-book/sitemap.xml

## Project Structure

### Documentation (this feature)

```text
specs/001-book-rag-ingestion/
├── plan.md              # This file - overall implementation plan
├── research.md          # Research findings and technical analysis
├── data-model.md        # Data structures and schemas
├── quickstart.md        # Quick setup and usage guide
├── contracts/           # API contracts and interface definitions
│   └── embedding-contract.md
└── tasks.md             # Detailed implementation tasks (created later)
```

### Source Code Structure

```text
backend/
├── pyproject.toml       # uv-managed project configuration
├── .env.example         # Environment variables template
├── .gitignore           # Python-specific gitignore
├── main.py              # Single file implementation with all required functions
├── test_ingestion.py    # Test suite for the ingestion system
├── README.md            # Project overview and usage
├── docs/
│   ├── architecture.md  # System architecture documentation
│   ├── data-model.md    # Data model documentation
│   └── quickstart.md    # Quick start guide
└── contracts/
    └── embedding-contract.md  # API contracts
```

**Structure Decision**: Implemented single-file approach as specified with main.py containing all required functions: get_all_urls, extract_text_from_url, chunk_text, embed, create_collection named rag_embedding, and save_chunk_to_qdrant, with a main function to execute the complete pipeline for https://sanilahmed.github.io/hackathon-ai-book/

## Implementation Phases

### Phase 1: Setup and Dependencies
- Create backend directory
- Initialize with uv package manager
- Set up pyproject.toml with required dependencies
- Create environment configuration

### Phase 2: Core Function Implementation
- Implement URL discovery function
- Create content extraction functionality
- Develop text chunking logic
- Implement Cohere embedding integration
- Create Qdrant storage functions

### Phase 3: Integration and Testing
- Integrate all functions in main execution flow
- Add error handling and logging
- Test with target book URL
- Validate data storage in Qdrant

## Risk Analysis and Mitigation

### Technical Risks
1. **API Rate Limits**: Mitigate with proper delays and batch processing
2. **Large Book Processing**: Implement memory-efficient processing
3. **Network Reliability**: Add retry mechanisms and timeout handling
4. **Content Extraction Failures**: Implement fallback selectors and error recovery

### Mitigation Strategies
- Add exponential backoff for API calls
- Implement progress tracking for large books
- Use efficient data structures to minimize memory usage
- Create comprehensive error logging for debugging

## Success Criteria

- Successfully discover all URLs from https://sanilahmed.github.io/hackathon-ai-book/
- Extract clean text content from each page
- Chunk content appropriately without losing context
- Generate embeddings using Cohere without errors
- Store all embeddings in Qdrant Cloud collection named "rag_embedding"
- Include proper metadata (URL, title, etc.) with each stored embedding
- Complete pipeline execution with proper logging and error handling

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Single file architecture | Requirement specified in user input | Modular approach would be more maintainable but not requested |