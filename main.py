import json
import sys
from typing import List
from haystack import Pipeline
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.components.converters import OpenAPIServiceToFunctions
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage, ByteStream

from components import (
    LLMJSONFormatEnforcer,
    OpenAIJSONGenerator,
    ChatMessageToJSONConverter,
    FormattedOutputProcessor,
    FormattedOutputTarget,
)

from util import (
    has_authentication_method,
    get_env_var_or_default,
    fetch_content_from,
    extract_custom_instruction,
    contains_skip_instruction,
    GITHUB_OUTPUT_TEMPLATE,
    STDOUT_OUTPUT_TEMPLATE,
)

if __name__ == "__main__":
    # make sure we have the required environment variables and have content for the required files
    get_env_var_or_default(env_var_name="OPENAI_API_KEY", required=True)
    open_api_spec = fetch_content_from(
        [get_env_var_or_default(env_var_name="OPENAPI_SERVICE_SPEC", required=True)], required=True
    )
    system_prompt_text = fetch_content_from(
        [get_env_var_or_default(env_var_name="SYSTEM_PROMPT", required=True)], required=True
    )
    function_calling_prompt = get_env_var_or_default(env_var_name="FUNCTION_CALLING_PROMPT", required=True)

    # and the optional environment variables
    text_generation_model_name = get_env_var_or_default(
        env_var_name="TEXT_GENERATION_MODEL", default_value="gpt-4-1106-preview"
    )
    function_calling_model_name = get_env_var_or_default(
        env_var_name="FUNCTION_CALLING_MODEL", default_value="gpt-3.5-turbo-0613"
    )
    github_output_file = get_env_var_or_default(env_var_name="GITHUB_OUTPUT")
    output_key = get_env_var_or_default(env_var_name="OUTPUT_KEY", default_value="text_generation")
    output_schema = get_env_var_or_default(env_var_name="OUTPUT_SCHEMA")
    output_schema = fetch_content_from([output_schema]) if output_schema else None
    bot_name = get_env_var_or_default(env_var_name="BOT_NAME")
    service_token = get_env_var_or_default(env_var_name="OPENAPI_SERVICE_TOKEN")
    user_prompt = get_env_var_or_default(env_var_name="USER_PROMPT")
    user_prompt = fetch_content_from([user_prompt]) if user_prompt else None
    service_response_subtree = get_env_var_or_default(env_var_name="SERVICE_RESPONSE_SUBTREE")
    user_instruction = extract_custom_instruction(bot_name, user_prompt) if user_prompt and bot_name else None

    if user_instruction and contains_skip_instruction(user_instruction):
        print("Exiting, user prompt contains the word 'skip'.")
        sys.exit(0)

    spec_to_functions = OpenAPIServiceToFunctions()
    openai_functions_definition_docs = spec_to_functions.run(
        sources=[ByteStream.from_string(text=open_api_spec)], system_messages=["TODO-REMOVE-ME"]
    )
    openai_functions_definition: ChatMessage = openai_functions_definition_docs["documents"][0]
    openai_functions_definition: str = json.loads(openai_functions_definition.content)

    open_api_service_spec_as_json = json.loads(open_api_spec)
    # Setup service authorization token, title is required field in OpenAPI spec
    service_title = open_api_service_spec_as_json["info"]["title"]
    if has_authentication_method(open_api_service_spec_as_json):
        if not service_token:
            raise ValueError(
                f"Service {service_title} requires authorization token. "
                f"Please set OPENAPI_SERVICE_TOKEN environment variable."
            )
        else:
            service_auth = {service_title: service_token}
    else:
        service_auth = None

    invoke_service_pipe = Pipeline()
    invoke_service_pipe.add_component("function_llm", OpenAIChatGenerator(model=function_calling_model_name))
    invoke_service_pipe.add_component("openapi_container", OpenAPIServiceConnector(service_auth))
    invoke_service_pipe.connect("function_llm.replies", "openapi_container.messages")

    resolve_service_params_messages = [
        ChatMessage.from_system("You are a helpful assistant capable of function calling."),
        ChatMessage.from_user(function_calling_prompt),
    ]

    tools_param = [{"type": "function", "function": openai_functions_definition}]
    tool_choice = {"type": "function", "function": {"name": openai_functions_definition["name"]}}

    service_response = invoke_service_pipe.run(
        data={
            "messages": resolve_service_params_messages,
            "generation_kwargs": {"tools": tools_param, "tool_choice": tool_choice},
            "service_openapi_spec": open_api_service_spec_as_json,
        }
    )

    service_response_msgs: List[ChatMessage] = service_response["openapi_container"]["service_response"]
    service_response_json = json.loads(service_response_msgs[0].content)
    service_response_json = (
        service_response_json[service_response_subtree] if service_response_subtree else service_response_json
    )
    diff_message = ChatMessage.from_user(json.dumps(service_response_json))

    system_message = ChatMessage.from_system(system_prompt_text)
    if user_instruction:
        text_gen_prompt_messages = [system_message] + [diff_message] + [ChatMessage.from_user(user_instruction)]
    else:
        text_gen_prompt_messages = [system_message] + [diff_message]

    gen_text_pipeline = Pipeline()
    routes = [
        {
            "condition": "{{has_output_schema}}",
            "output": "{{messages}}",
            "output_name": "needs_fc",
            "output_type": List[ChatMessage],
        },
        {
            "condition": "{{not has_output_schema}}",
            "output": "{{messages}}",
            "output_name": "no_fc",
            "output_type": List[ChatMessage],
        },
    ]

    gen_text_pipeline.add_component(
        "llm", OpenAIChatGenerator(model=text_generation_model_name, generation_kwargs={"max_tokens": 2560})
    )
    gen_text_pipeline.add_component("post", LLMJSONFormatEnforcer())
    gen_text_pipeline.add_component("router", ConditionalRouter(routes))
    gen_text_pipeline.add_component("fc_llm", OpenAIJSONGenerator())
    gen_text_pipeline.add_component("msg_to_json", ChatMessageToJSONConverter())
    gen_text_pipeline.add_component("github_output", FormattedOutputProcessor())
    gen_text_pipeline.connect("llm.replies", "post.messages")
    gen_text_pipeline.connect("post.messages", "router")
    gen_text_pipeline.connect("router.needs_fc", "fc_llm.messages")
    gen_text_pipeline.connect("router.no_fc", "msg_to_json.messages")
    gen_text_pipeline.connect("msg_to_json.output", "github_output.json_objects")
    gen_text_pipeline.connect("fc_llm.output", "github_output.json_objects")

    output_sinks = [FormattedOutputTarget(github_output_file, GITHUB_OUTPUT_TEMPLATE)] if github_output_file else []
    # always output to stdout for debugging unless quiet mode is enabled
    if not get_env_var_or_default(env_var_name="QUIET", default_value=False):
        output_sinks.append(FormattedOutputTarget(sys.stdout, STDOUT_OUTPUT_TEMPLATE))

    gen_text_pipeline.run(
        data={
            "messages": text_gen_prompt_messages,
            "has_output_schema": bool(output_schema),
            "output_key": output_key,
            "output_schema": output_schema,
            "output_targets": output_sinks,
        }
    )
