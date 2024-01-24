import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import List, Tuple, Dict, Any
from typing import Optional
from urllib.parse import urlparse

import requests
from haystack import Pipeline
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.components.converters import OpenAPIServiceToFunctions
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ByteStream


def get_env_var_or_default(env_var_name: str, default_value: Optional[str] = None, required: bool = False) -> str:
    """
    Retrieves the value of an environment variable if it exists, otherwise returns a default value or raises an
    error if required.

    This function fetches the value of an environment variable specified by `env_var_name`. If the environment
    variable does not exist, it either returns a `default_value` if provided, or raises a `ValueError` if the
    `required` flag is set to `True`. If `default_value` is not provided and `required` is `False`, it returns `None`.

    :param env_var_name: The name of the environment variable to retrieve.
    :type env_var_name: str
    :param default_value: The default value to return if the environment variable is not found. Defaults to `None`.
    :type default_value: Optional[str]
    :param required: If set to `True`, the function raises a `ValueError` when the environment variable is not found. Defaults to `False`.
    :type required: bool
    :return: The value of the environment variable or the `default_value`. Returns `None` if the environment variable is not found and `default_value` is `None`.
    :rtype: str
    :raises ValueError: If `required` is `True` and the environment variable is not found.
    """
    env_var_value = os.environ.get(env_var_name)

    if env_var_value is not None:
        return env_var_value
    elif default_value is not None:
        return default_value
    elif required:
        raise ValueError(f"Please set {env_var_name} environment variable.")

    return None


def is_valid_url(location: str):
    """
    Checks if the location is a valid URL.
    :param location: a string representing a URL
    :return: True if the location is a valid URL; otherwise, False.
    """
    parsed = urlparse(location)
    return bool(parsed.netloc)


def load_text_from(locations: List[str]) -> Tuple[bool, str]:
    """
    Attempts to load text from a list of locations.

    Each location can be a URL, a file path, or an environment variable name.
    Returns a tuple (success, text or error message).

    :param locations: List of strings representing URLs, file paths, or environment variable names.
    :return: Tuple containing a boolean indicating success, and the loaded text or error message.
    """
    for location in locations:
        try:
            if is_valid_url(location):
                return load_from_url(location)

            elif os.path.exists(location):
                return load_from_file(location)

            elif location in os.environ:
                return True, os.environ[location]

            else:
                return True, location

        except Exception as e:
            logging.error(f"Failed to load text from {location}. Error: {e}")

    return False, "Failed to load from any of the locations."


def load_from_url(url: str) -> Tuple[bool, str]:
    response = requests.get(url)
    if response.status_code == 200:
        return True, response.text
    else:
        return False, f"Failed to load from URL: {url} with status code: {response.status_code}"


def load_from_file(file_path: str) -> Tuple[bool, str]:
    with open(file_path, "r") as file:
        return True, file.read()


def has_authentication_method(openapi_service_json: Dict[str, Any]) -> bool:
    """
    Checks if the OpenAPI service specification contains an authentication method.
    If so, it is assumed that endpoint requires authentication.

    :param openapi_service_json: The OpenAPI service specification in JSON format.
    :type openapi_service_json: Dict[str, Any]
    :return: True if the service specification contains an authentication method; otherwise, False.
    :rtype: bool
    """
    return "components" in openapi_service_json and "securitySchemes" in openapi_service_json["components"]


def text_generate(
    text_model: str,
    function_model: str,
    function_model_prompt: str,
    open_api_service_spec: str,
    service_response_root: Optional[str] = None,
    custom_instruction: Optional[str] = None,
) -> ChatMessage:
    """
    Orchestrates LLM text generation by leveraging a Haystack OpenAPI service functionality. It first converts an
    OpenAPI service specification into a set of OpenAI callable functions, sets up service authorization, and then
    invokes OpenAPI service injecting service response into a prompt to generate LLM response based on user prompts
    and custom user instructions.

    It's essential to ensure that the `open_api_service_spec` is validated against OpenAPI 3.x schema and properly
    in formatted JSON string format.

    :param str text_model: The identifier of the text model used for generating the final text.
    :param str function_model: The identifier of the function calling model used to resolve parameters for
    OpenAPI service invocation
    :param str function_model_prompt: The user prompt for the function calling
    model e.g (What's the weather like in Berlin for the next three days?)
    :param str open_api_service_spec: The OpenAPI service specification in string JSON format.
    :param Optional[str] service_response_root: Optional element to extract from the service response. Default is None.
    :param Optional[str] custom_instruction: Additional custom user instruction for text generation. Default is None.
    :return: The generated `ChatMessage` based on the provided inputs and interactions with the specified services.
    :rtype: ChatMessage
    """

    spec_to_functions = OpenAPIServiceToFunctions()
    openai_functions_definition_docs = spec_to_functions.run(
        sources=[ByteStream.from_string(text=open_api_service_spec)], system_messages=["TODO-REMOVE-ME"]
    )
    openai_functions_definition: ChatMessage = openai_functions_definition_docs["documents"][0]
    openai_functions_definition: str = json.loads(openai_functions_definition.content)

    open_api_service_spec_as_json = json.loads(open_api_service_spec)
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
    invoke_service_pipe.add_component("function_llm", OpenAIChatGenerator(model=function_model))
    invoke_service_pipe.add_component("openapi_container", OpenAPIServiceConnector(service_auth))
    invoke_service_pipe.connect("function_llm.replies", "openapi_container.messages")

    resolve_service_params_messages = [
        ChatMessage.from_system("You are a helpful assistant capable of function calling."),
        ChatMessage.from_user(function_model_prompt),
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
        service_response_json[service_response_root] if service_response_root else service_response_json
    )
    diff_message = ChatMessage.from_user(json.dumps(service_response_json))

    system_message = ChatMessage.from_system(system_prompt_text)
    if custom_instruction:
        text_gen_prompt_messages = [system_message] + [diff_message] + [ChatMessage.from_user(custom_instruction)]
    else:
        text_gen_prompt_messages = [system_message] + [diff_message]

    gen_text_pipeline = Pipeline()
    # TODO: perhaps provide env var for max_tokens, 2560 should be enough for many cases
    # Note that you can use OPENAI_ORG_ID to set the organization ID for your OpenAI API key to track usage and costs
    llm = OpenAIChatGenerator(model=text_model, generation_kwargs={"max_tokens": 2560})
    gen_text_pipeline.add_component("llm", llm)

    final_result = gen_text_pipeline.run(data={"messages": text_gen_prompt_messages})
    return final_result["llm"]["replies"][0]


def extract_custom_instruction(bot_name: str, user_instruction: str) -> str:
    """
    Extracts custom instruction from a user instruction string by searching for specific pattern in the user
    instruction string to find and return custom instructions.

    The function uses regular expressions to find the custom instruction following the bot name in the user instruction

    :param bot_name: The name of the bot to search for in the user instruction string.
    :type bot_name: str
    :param user_instruction: The complete user instruction string, potentially containing custom instructions.
    :type user_instruction: str
    :return: The extracted custom instruction, if found; otherwise, an empty string.
    :rtype: str
    """
    # Search for the message following @bot_name
    match = re.search(rf"@{re.escape(bot_name)}\s+(.*)", user_instruction)
    return match.group(1) if match else ""


def contains_skip_instruction(text):
    return bool(re.search(r"\bskip\b", text, re.IGNORECASE))


def write_to_output(file_handle: str, json_input: str):
    """
     Writes the contents of a JSON object to a file, formatted for compatibility with GitHub Actions. This function
     is specifically designed to handle the 'outputs' key within the JSON object and write each output value to the
     specified file. It ensures that multiline outputs are handled correctly, as GitHub Actions does not support
     multiline strings in outputs directly.

    :param str file_handle: The file path or handle where the output is to be written.
    :param str json_input: A JSON string containing the outputs to be written to the file.
    """

    # multiple lines outputs are not supported in GitHub Actions
    # see https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#multiline-strings
    # therefore, we need to use a unique delimiter to mark the output boundaries
    delimiter = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]

    # iterate over "outputs" key in the JSON object and write each output to file
    try:
        json_object = json.loads(json_input)
    except Exception:
        logging.error(f"LLM failed to generate well formed JSON response: {json_input}")
        logging.warning(f"Skipping writing the JSON response to file: {file_handle}")
        return

    if "outputs" not in json_object:
        logging.warning(f"LLM failed to generate JSON response with the expected 'outputs' key: {json_input}")
        logging.warning(f"Writing the entire JSON response to file: {file_handle}")

        with open(file_handle, "a") as env_file:
            for key, value in json_object.items():
                env_file.write(f"{key}<<{delimiter}\n")
                env_file.write(f"{value}\n")
                env_file.write(f"{delimiter}\n")

    for output_name, output_value in json_object["outputs"].items():
        with open(file_handle, "a") as env_file:
            env_file.write(f"{output_name}<<{delimiter}\n")
            env_file.write(f"{output_value}\n")
            env_file.write(f"{delimiter}\n")


def is_valid_json(json_string: str) -> bool:
    """
    Checks if a string is a valid JSON object.
    :param json_string: a string representing a JSON object
    :return: True if the string is a valid JSON object; otherwise, False.
    """
    try:
        json_object = json.loads(json_string)
        return isinstance(json_object, dict)
    except Exception:
        return False


def post_process(message: ChatMessage, output_json_key: str) -> ChatMessage:
    """
    Post-processes the generated message
    :param message: The generated message to be post-processed.
    :type message: ChatMessage
    :param output_json_key: The key to use for the output of the generated message.
    :type output_json_key: str
    """
    if not is_valid_json(message.content):
        resp = message.content
        # remove various ```code or ```json code callouts LLMs may generate
        resp = resp.replace("`", "").replace("`", "").strip()
        resp = resp.replace("code", "").strip()
        resp = resp.replace("json", "").strip()
        resp = resp.replace("markdown", "").strip()

        # call json.dumps to remove all characters that are not valid JSON,
        # dump the response into a JSON object under predefined key
        message.content = '{"outputs":{"' + output_json_key + '":' + json.dumps(resp) + "}}"
    return message


if __name__ == "__main__":
    # make sure we have the required environment variables
    get_env_var_or_default(env_var_name="OPENAI_API_KEY", required=True)
    load_ok, open_api_spec = load_text_from(
        [get_env_var_or_default(env_var_name="OPENAPI_SERVICE_SPEC", required=True)]
    )
    if not load_ok:
        print("Exiting, failed to load OpenAPI service specification.")
        sys.exit(0)
    load_ok, system_prompt_text = load_text_from([get_env_var_or_default(env_var_name="SYSTEM_PROMPT", required=True)])
    if not load_ok:
        print("Exiting, failed to load system prompt text.")
        sys.exit(0)
    function_calling_prompt = get_env_var_or_default(env_var_name="FUNCTION_CALLING_PROMPT", required=True)

    # and the optional ones
    text_generation_model_name = get_env_var_or_default(
        env_var_name="TEXT_GENERATION_MODEL", default_value="gpt-4-1106-preview"
    )
    function_calling_model_name = get_env_var_or_default(
        env_var_name="FUNCTION_CALLING_MODEL", default_value="gpt-3.5-turbo-0613"
    )
    output_file = get_env_var_or_default(env_var_name="OUTPUT_FILE", default_value="GITHUB_OUTPUT")
    output_key = get_env_var_or_default(env_var_name="OUTPUT_KEY", default_value="output")
    bot_name = get_env_var_or_default(env_var_name="BOT_NAME")
    service_token = get_env_var_or_default(env_var_name="OPENAPI_SERVICE_TOKEN")
    user_prompt = get_env_var_or_default(env_var_name="USER_PROMPT")
    _, user_prompt = load_text_from([user_prompt]) if user_prompt else (False, None)
    service_response_subtree = get_env_var_or_default(env_var_name="SERVICE_RESPONSE_SUBTREE")
    user_instruction = extract_custom_instruction(bot_name, user_prompt) if user_prompt and bot_name else None

    if user_instruction and contains_skip_instruction(user_instruction):
        print("Exiting, user prompt contains the word 'skip'.")
        sys.exit(0)

    # Ok, we are gtg, generate text
    generated_message = text_generate(
        text_model=text_generation_model_name,
        function_model=function_calling_model_name,
        function_model_prompt=function_calling_prompt,
        open_api_service_spec=open_api_spec,
        service_response_root=service_response_subtree,
        custom_instruction=user_instruction,
    )

    generated_message = post_process(generated_message, output_key)
    meta_info = '{"outputs":{"generation_stats":' + json.dumps(generated_message.meta) + "}}"

    # output the generated text and the generation statistics to the console (i.e. for docker experiments)
    print(f"{generated_message.content}\n{meta_info}")

    # write the output to file as JSON, each output has single outputs JSON key
    write_to_output(output_file, generated_message.content)
    write_to_output(output_file, meta_info)
