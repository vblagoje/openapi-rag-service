import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Union, TextIO
from typing import Optional
from urllib.parse import urlparse

import requests
from haystack import Pipeline
from haystack import component
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.components.converters import OpenAPIServiceToFunctions
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.core.component.types import Variadic
from haystack.dataclasses import ByteStream
from haystack.dataclasses import ChatMessage
from jinja2 import Template
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

GITHUB_OUTPUT_TEMPLATE = """
{{output_name}}<<{{delimiter()}}
{{render(output_value)}}
{{delimiter()}}
"""

STDOUT_OUTPUT_TEMPLATE = """
{{output_name}}:
{{render(output_value)}}
"""


# ----------- util.py ----------------


def ensure_json_objects(data):
    """
    Recursively traverses a dictionary, converting any string values
    that are valid JSON objects into dictionary objects.

    :param data: The dictionary to be traversed and modified.
    :return: The modified dictionary with JSON strings converted to dictionary objects.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")

    for key, value in data.items():
        if isinstance(value, str):
            try:
                # Attempt to load the string as JSON
                json_value = json.loads(value)
                if isinstance(json_value, dict):
                    # If the loaded JSON is a dictionary, recursively process it
                    data[key] = ensure_json_objects(json_value)
                else:
                    # If the loaded JSON is not a dictionary, just update the value
                    data[key] = json_value
            except json.JSONDecodeError:
                # If json.loads fails, leave the value as is
                pass
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively process it
            data[key] = ensure_json_objects(value)

    return data


def get_env_var_or_default(
        env_var_name: str, default_value: Optional[Any] = None, required: bool = False
) -> Optional[str]:
    """
    Retrieves the value of an environment variable if it exists, otherwise returns a default value or raises an
    error if required.

    :param env_var_name: The name of the environment variable to retrieve.
    :param default_value: The default value to return if the environment variable is not found. Defaults to `None`.
    :param required: If set to `True`, the function raises a `ValueError` when the environment variable is not found.
    Defaults to `False`.
    :return: The value of the environment variable or the `default_value`. Returns `None` if the environment variable
    is not found and `default_value` is `None`.
    :raises ValueError: If `required` is `True` and the environment variable is not found.
    """
    env_var_value = os.environ.get(env_var_name)

    if env_var_value is not None:
        return env_var_value
    if default_value is not None:
        return default_value
    if required:
        raise ValueError(f"The required environment variable '{env_var_name}' is not set.")


def is_valid_url(location: str):
    """
    Checks if the location is a valid URL.
    :param location: a string representing a URL
    :return: True if the location is a valid URL; otherwise, False.
    """
    parsed = urlparse(location)
    return bool(parsed.netloc)


def fetch_content_from(locations: List[str], required: Optional[bool] = False) -> Optional[str]:
    """
    Attempts to load content from a given list of locations. These locations can be URLs, file paths,
    or environment variable names. The function tries each location in order and returns the content
    from the first successful source.

    If all attempts fail, a ValueError is raised unless `required` is set to `False`.

    :param locations: A list of strings representing the locations to try loading text from. Each location
                      can be a URL, a file path, or an environment variable name.
    :param required: If set to `True`, the function raises a `ValueError` when the text cannot be loaded from
                        any of the provided locations. Defaults to `False`.
    :return: The text content from the first successful location.
    :raises ValueError: If none of the locations result in successful text retrieval, indicating failure
                        to load text from any provided location.
    :raises requests.exceptions.RequestException: If an error occurs while fetching content from a URL.
    :raises IOError: If an error occurs while reading from a file.

    Example:
        locations = ["https://example.com/data.txt", "local_data.txt", "DATA_ENV_VAR"]
        try:
            content = load_text_from(locations)
            print(content)
        except ValueError:
            print("Failed to load content from any location.")
    """
    for location in locations:
        try:
            if is_valid_url(location):
                return load_from_url(location)

            elif os.path.exists(location):
                return load_from_file(location)

            elif location in os.environ:
                return os.environ[location]

        except Exception as e:
            logging.error(f"Failed to load text from {location}: {e}")
            continue

    if required:
        raise ValueError(f"Failed to load text from any of the provided locations {locations}")
    return None


def load_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError for non-200 responses
        return response.text
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to load from URL: {url}. Error: {e}")


def load_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise ValueError(f"Failed to load from file: {file_path}. Error: {e}")


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


# ----------- components.py ----------------


@dataclass
class FormattedOutputTarget:
    """
    A class representing an output target with formatting instructions.

    Attributes:
        destination (Union[str, TextIO]): The output destination, which can be a file path (str) or a file-like (TextIO)
        template (str): A string template for formatting the output.
    """

    destination: Union[str, TextIO]
    template: str

    def get_file_handle(self, mode: str = "a"):
        """
        Returns a file handle to the destination. If the destination is a string (file path), it opens the file in
        the specified mode. If it's a file-like object, it returns a null context for safe usage in a 'with' statement.

        :param mode: The mode in which to open the file. Default is 'a' (append mode).
        :return: A context manager managing the file handle.
        """
        if isinstance(self.destination, str):
            try:
                return open(self.destination, mode)
            except IOError as e:
                logging.error(f"Failed to open file {self.destination}: {e}")
                raise
        else:
            return contextlib.nullcontext(self.destination)


@component
class LLMJSONFormatEnforcer:
    """
    This class enforces the JSON format for the output of a Large Language Model (LLM) to ensure
    compatibility and standardization. It primarily modifies the last message in a list of ChatMessage
    instances to ensure its content is valid JSON, wrapping it in a JSON object if necessary.
    """

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage], output_key: str):
        """
        Processes the last message in a list of ChatMessage instances, ensuring that its content is
        valid JSON. If the content is not valid JSON, it attempts to format it as JSON using a specified
        output key.

        :param messages: A list of ChatMessage instances. The last message in this list is processed.
        :param output_key: The key used to wrap the content if it's not already valid JSON.
        :return: A dictionary containing the list of messages under the key 'messages'.
        """
        message = messages[-1]
        if not self.is_valid_json(message.content):
            # Attempt to remove the outermost code block and check for valid JSON again
            resp = self.remove_outer_code_blocks(message.content)
            message.content = json.dumps({output_key: resp}) if not self.is_valid_json(resp) else resp

        return {"messages": messages}

    def remove_outer_code_blocks(self, text: str):
        """
        Strips the outermost code block delimiters (triple backticks) from a given string, if present.
        This is used to extract the core content from formatted text blocks.

        :param text: The string from which to remove the outer code blocks.
        :return: The processed string with outer code blocks removed, if they were present.
        """
        text = text.strip()

        # Check if the text starts with and ends with callouts
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n") + 1
            last_callout_start = text.rfind("```")
            # Slice the text to remove the first and last callouts
            return text[first_newline:last_callout_start].strip()
        else:
            return text

    def is_valid_json(self, json_string: str) -> bool:
        """
        Checks whether a given string is valid JSON. Specifically, it verifies if the string can be
        successfully parsed into a JSON object (dict in Python).

        :param json_string: The string to check for JSON validity.
        :return: True if the string is valid JSON; False otherwise.
        """
        try:
            json_object = json.loads(json_string)
            return isinstance(json_object, dict)
        except json.JSONDecodeError:
            return False


@component
class OpenAIJSONGenerator:
    """
    OpenAIJSONGenerator interfaces with the OpenAI API and compatible LLM providers to enforce a JSON-constrained
    output from a Large Language Model (LLM). This class is guided by a function calling schema, originally defined
    by OpenAI, but now also supported by various other LLM providers. It leverages this function calling mechanism
    to process and format LLM outputs according to the specified schema, ensuring that the generated responses
    adhere to the desired structure and content format. This makes the class versatile for use with multiple LLM
    platforms that support the OpenAI Python client and its function calling conventions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo-0613",
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.api_base_url = api_base_url
        self.organization = organization
        self.client = OpenAI(api_key=api_key, organization=organization, base_url=api_base_url)
        """
        Initializes the OpenAIJSONGenerator with API key, model, base URL, organization, and any additional arguments.

        :param api_key: Optional; The API key for accessing the OpenAI service.
        :param model: The model name to be used with the OpenAI API. Default is 'gpt-3.5-turbo-0613'.
        :param api_base_url: Optional; The base URL for the OpenAI API.
        :param organization: Optional; The organization ID for the OpenAI account.
        :param generation_kwargs: Optional; Additional keyword arguments for the OpenAI generation call.
        """

    @component.output_types(output=List[Dict[str, Any]])
    def run(self, messages: List[ChatMessage], output_schema: str):
        """
        Generates JSON responses based on the given ChatMessages and the specified output schema. It sends the
        formatted messages to the OpenAI API and processes the response according to the output schema.

        :param messages: A list of ChatMessage instances to process.
        :param output_schema: A JSON string defining the schema for the output.
        :return: A dictionary containing the processed responses under the key 'output'.
        """
        output_schema_json = json.loads(output_schema)
        format_to_json_message = [messages[-1]]

        tools_param = [{"type": "function", "function": output_schema_json}]
        tool_choice = {"type": "function", "function": {"name": output_schema_json["name"]}}
        generation_kwargs = {"tools": tools_param, "tool_choice": tool_choice}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = self._convert_to_openai_format(format_to_json_message)

        chat_completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore # openai expects list of specific message types
            stream=False,
            **generation_kwargs,
        )

        completions: List[Dict[str, Any]] = self._build_response(chat_completion)
        fn_calls = [ensure_json_objects(completion)["function"]["arguments"] for completion in completions]
        return {"output": fn_calls}

    def _convert_to_openai_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Converts a list of ChatMessage instances into a format suitable for the OpenAI API.

        :param messages: A list of ChatMessage instances.
        :return: A list of dictionaries formatted for the OpenAI API.
        """
        openai_chat_message_format = {"role", "content", "name"}
        openai_formatted_messages = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            filtered_message = {k: v for k, v in message_dict.items() if k in openai_chat_message_format and v}
            openai_formatted_messages.append(filtered_message)
        return openai_formatted_messages

    def _build_response(self, completion: ChatCompletion) -> List[Dict[str, Any]]:
        """
        Constructs the final response from the OpenAI API completion result. It formats the response
        based on whether it includes function calls or tool calls.

        :param completion: The ChatCompletion object received from the OpenAI API.
        :return: A list of dictionaries representing the formatted response.
        """
        message: ChatCompletionMessage = completion.choices[0].message
        json_content: List[Dict[str, Any]] = []
        if message.function_call:
            # here we mimic the tools format response so that if user passes deprecated `functions` parameter
            # she'll get the same output as if new `tools` parameter was passed
            # use pydantic model dump to serialize the function call
            json_content = [
                {"function": message.function_call.model_dump(mode="json"), "type": "function", "id": completion.id}
            ]

        elif message.tool_calls:
            # new `tools` parameter was passed, use pydantic model dump to serialize the tool calls
            json_content = [tc.model_dump(mode="json") for tc in message.tool_calls]

        else:
            raise ValueError(f"Received no tools response from OpenAI API.{message}")

        return json_content


@component
class ChatMessageToJSONConverter:
    """
    Converts a list of ChatMessage instances to a list of JSON objects. This conversion
    assumes that the 'content' field of each ChatMessage is a valid JSON string. Messages
    that do not contain valid JSON are logged and skipped.
    """

    @component.output_types(output=List[Dict[str, Any]])
    def run(self, messages: List[ChatMessage]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Converts each ChatMessage in the provided list to a JSON object, if possible.

        :param messages: A list of ChatMessage instances to be converted.
        :return: A dictionary containing the list of successfully converted JSON objects under the key 'output'.
        """
        json_output = []
        for message in messages:
            try:
                json_content = json.loads(message.content)
                json_output.append(json_content)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in message: {message.content}, skipping.")

        return {"output": json_output}


@component
class FormattedOutputProcessor:
    """
    Processes and writes formatted output from JSON objects to specified output targets.
    This class is designed to handle multiple JSON objects and write them to various output
    destinations, which can include files or standard outputs, using specified formatting templates.
    """

    @component.output_types(output=List[Dict[str, Any]])
    def run(self, json_objects: Variadic[List[Dict[str, Any]]], output_targets: List[FormattedOutputTarget]):
        """
        Iterates over a collection of JSON objects and output targets, writing each JSON object
        to the specified targets with appropriate formatting.

        :param json_objects: A nested list of JSON objects to be processed and written to the output targets.
        :param output_targets: A list of FormattedOutputTarget instances specifying where and how to write the outputs.
        :return: A dictionary containing the processed JSON objects under the key 'output'.
        """
        for json_object in itertools.chain(*json_objects):
            for output_target in output_targets:
                self.write_formatted_output(json_object, output_target)

        return {"output": json_objects}

    def create_unique_delimiter(self):
        """
        Generates a unique delimiter string based on the current time. This is used in formatting
        the output to ensure uniqueness in the delimiter used in the output templates.

        :return: A string representing a unique delimiter.
        """
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]

    def fetch_output_formatter(self, output_name: str):
        """
        Retrieves a formatter function for a given output name. If a specific template is defined
        for the output name, the function will format the output accordingly. Otherwise, it returns
        the output as is.

        :param output_name: The name of the output for which to fetch the formatter.
        :return: A function that takes a value and returns a formatted output.
        """
        template_name = get_env_var_or_default(env_var_name=output_name.upper() + "_TEMPLATE")
        template_str = fetch_content_from([template_name]) if template_name else None

        if template_str:
            template = Template(template_str)
            # Return a function that renders json with the template
            return lambda value: template.render(**value)
        else:
            # If no template is found, return the original value as is
            return lambda value: value

    def write_formatted_output(self, data_to_write: Dict[str, Any], file_target: FormattedOutputTarget):
        """
        Writes a given data object to the specified output target using the provided template.
        The output is formatted based on the template associated with each output name within the data.

        :param data_to_write: A dictionary representing the data to be written.
        :param file_target: A FormattedOutputTarget instance specifying where and how to write the data.
        """
        try:
            template = Template(file_target.template)
            for output_name, output_value in data_to_write.items():
                output_value_renderer = self.fetch_output_formatter(output_name)
                delimiter_value = self.create_unique_delimiter()  # Generate once for each output
                formatted_output = template.render(
                    output_name=output_name,
                    output_value=output_value,
                    delimiter=lambda: delimiter_value,  # Pass the generated value
                    render=output_value_renderer,
                )
                with file_target.get_file_handle() as sink:
                    sink.write(formatted_output + "\n")
        except Exception as e:
            logging.error(f"Error processing JSON object: {data_to_write}")
            logging.error(f"Exception: {e}")


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
