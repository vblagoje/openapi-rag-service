# vblagoje/openapi-rag-service

`vblagoje/openapi-rag-service` demonstrates the seamless integration of OpenAPI-defined services with Large Language Models (LLMs) in the Haystack RAG (Retriever-Augmented Generation) pipeline. This project offers an easy way to explore the enhanced capabilities of Haystack's 2.x RAG architecture, enriching LLMs with structured data from various OpenAPI services. The vblagoje/openapi-rag-service Docker image facilitates quick experimentation, opening new possibilities for enriching LLM outputs in the RAG framework.
## Key Features

- **Seamless OpenAPI Integration**: Incorporates any OpenAPI-specified service into the RAG pipeline, expanding the potential sources of information beyond unstructured text.
- **Enhanced RAG Functionality**: Enhances traditional RAG capabilities with structured, service-driven data, offering more contextually rich and accurate outputs.
- **Flexible LLM Compatibility**: Supports various Large Language Models (LLMs) that adhere to OpenAI Python client standards, ensuring broad applicability and flexibility.

## Getting Started

### Prerequisites

- Docker installed on your machine.
- Access to LLM providers compatible with the OpenAI Python client.

### Running the Service

To run the `vblagoje/openapi-rag-service`, pull the Docker image and execute it with the necessary parameters:

```sh
docker pull vblagoje/openapi-rag-service:latest
docker run <additional-parameters> vblagoje/openapi-rag-service
```

## Configuration

Configure the service using the following environment variables:

- `OPENAI_API_KEY`: Your API key for OpenAI.
- `OPENAI_BASE_URL`: The base URL for the OpenAI API.
- `TEXT_GENERATION_MODEL`: Name of the model used for text generation (e.g., `gpt-4-1106-preview`).
- `FUNCTION_CALLING_MODEL`: Model name for handling function calls (e.g., `gpt-3.5-turbo-0613`).
- `SYSTEM_PROMPT`: System message or prompt URL to assist the model in generating content.
- `USER_PROMPT`: Additional user-defined prompt for content generation.
- `BOT_NAME`: Bot name used in guiding the generation process.
- `OPENAPI_SERVICE_SPEC`: URL or path to the OpenAPI service specification.
- `OPENAPI_SERVICE_TOKEN`: Token for authenticating with the specified OpenAPI service.
- `SERVICE_RESPONSE_SUBTREE`: Specific subtree to parse in the service response.
- `OUTPUT_KEY`: Key identifier for the output generation process.
- `OUTPUT_SCHEMA`: Schema URL or text defining the output format.
- `QUIET`: Set to `true` to disable output to standard output (STDOUT).

## Usage Example

TODO


## License

Licensed under (LICENCE)[LICENCSE]
