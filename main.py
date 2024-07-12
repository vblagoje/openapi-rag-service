import contextlib
import copy
import hashlib
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
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatRole
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_experimental.components.tools.openapi import OpenAPITool, LLMProvider

from jinja2 import Template

GITHUB_OUTPUT_TEMPLATE = """
{{output_name}}<<{{delimiter()}}
{{render(output_value)}}
{{delimiter()}}
"""

STDOUT_OUTPUT_TEMPLATE = """
{{output_name}}:
{{render(output_value)}}
"""

# ----------- filters and routers ----------------


def change_role_to_user(messages: List[ChatMessage]):
    messages[-1].role = ChatRole.USER
    messages[-1].meta = None
    return messages


def prepare_fc_params(openai_functions_schema: Dict[str, Any]) -> Dict[str, Any]:
    if openai_functions_schema:
        return {
            "tools": [{"type": "function", "function": openai_functions_schema}],
            "tool_choice": {
                "type": "function",
                "function": {"name": openai_functions_schema["name"]},
            },
        }
    else:
        return {}


cf = {
    "json_loads": lambda s: json.loads(s) if isinstance(s, str) else json.loads(str(s)),
    "change_role": change_role_to_user,
    "prepare_fc_params": prepare_fc_params,
}


def gen_text_routes():
    return [
        {
            "condition": "{{has_output_schema}}",
            "output": "{{messages}}",
            "output_name": "needs_function_calling",
            "output_type": List[ChatMessage],
        },
        {
            "condition": "{{not has_output_schema}}",
            "output": "{{messages}}",
            "output_name": "no_need_for_function_calling",
            "output_type": List[ChatMessage],
        },
    ]


def invoke_service_routes():
    return [
        {
            "condition": "{{has_json_schema}}",
            "output": "{{messages}}",
            "output_name": "with_error_correction",
            "output_type": List[ChatMessage],
        },
        {
            "condition": "{{not has_json_schema}}",
            "output": "{{messages}}",
            "output_name": "no_error_correction",
            "output_type": List[ChatMessage],
        },
    ]


# ----------- util.py ----------------


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
        raise ValueError(
            f"The required environment variable '{env_var_name}' is not set."
        )
    return None


def is_valid_url(location: str):
    """
    Checks if the location is a valid URL.
    :param location: a string representing a URL
    :return: True if the location is a valid URL; otherwise, False.
    """
    parsed = urlparse(location)
    return bool(parsed.netloc)


def fetch_content_from(
    locations: List[str], required: Optional[bool] = False
) -> Optional[str]:
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
        raise ValueError(
            f"Failed to load text from any of the provided locations {locations}"
        )
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

    destination: str
    template: str

    def get_file_handle(self, mode: str = "a"):
        """
        Returns a file handle to the destination. If the destination is a string (file path), it opens the file in
        the specified mode. If it's a file-like object, it returns a null context for safe usage in a 'with' statement.

        :param mode: The mode in which to open the file. Default is 'a' (append mode).
        :return: A context manager managing the file handle.
        """

        if self.destination == "stdout":
            return contextlib.redirect_stdout(sys.stdout)
        try:
            return open(self.destination, mode)
        except IOError as e:
            logging.error(f"Failed to open file {self.destination}: {e}")
            raise


@component
class OpenAPIServiceResponseTextGeneration:
    def run(
        self,
        openapi_service_response: List[ChatMessage],
        subtree: str,
        system_prompt: str,
        user_prompt: Optional[str] = None,
    ):
        service_response_msg = json.loads(openapi_service_response[0].content)
        service_response_msg = (
            service_response_msg[subtree] if subtree else service_response_msg
        )

        service_response_message = ChatMessage.from_user(
            json.dumps(service_response_msg)
        )
        sys_message = ChatMessage.from_system(system_prompt)
        if user_prompt:
            text_gen_prompt = (
                [sys_message]
                + [service_response_message]
                + [ChatMessage.from_user(user_prompt)]
            )
        else:
            text_gen_prompt = [sys_message] + [service_response_message]

        return {"prompt_messages": text_gen_prompt}


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
        last_message = messages[-1]
        modified_message = copy.deepcopy(last_message)
        if not self.is_valid_json(modified_message.content):
            # Attempt to remove the outermost code block and check for valid JSON again
            resp = self.remove_outer_code_blocks(modified_message.content)
            modified_message.content = (
                json.dumps({output_key: resp}) if not self.is_valid_json(resp) else resp
            )

        # Replace the last message with the modified copy
        messages[-1] = modified_message
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
class FormattedOutputProcessor:
    """
    Processes and writes formatted output from JSON objects to specified output targets.
    This class is designed to handle multiple JSON objects and write them to various output
    destinations, which can include files or standard outputs, using specified formatting templates.
    """

    @component.output_types(output=List[Dict[str, Any]])
    def run(
        self, json_object: Dict[str, Any], output_targets: List[FormattedOutputTarget]
    ):
        """
        Iterates over a collection of JSON objects and output targets, writing each JSON object
        to the specified targets with appropriate formatting.

        :param json_object: A dictionary representing the JSON object to be written.
        :param output_targets: A list of FormattedOutputTarget instances specifying where and how to write the outputs.
        :return: A dictionary containing the processed JSON objects under the key 'output'.
        """
        for output_target in output_targets:
            self.write_formatted_output(json_object, output_target)
        return {"output": json_object}

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
        template_name = get_env_var_or_default(
            env_var_name=output_name.upper() + "_TEMPLATE"
        )
        template_str = fetch_content_from([template_name]) if template_name else None

        if template_str:
            template = Template(template_str)
            # Return a function that renders json with the template
            return lambda value: template.render(**value)
        else:
            # If no template is found, return the original value as is
            return lambda value: value

    def write_formatted_output(
        self, data_to_write: Dict[str, Any], file_target: FormattedOutputTarget
    ):
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
                delimiter_value = (
                    self.create_unique_delimiter()
                )  # Generate once for each output
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
    get_env_var_or_default("OPENAI_API_KEY", required=True)
    open_api_spec_url = get_env_var_or_default("OPENAPI_SERVICE_SPEC", required=True)
    system_prompt_text = fetch_content_from(
        [get_env_var_or_default("SYSTEM_PROMPT", required=True)],
        required=True,
    )
    function_calling_prompt = get_env_var_or_default(env_var_name="FUNCTION_CALLING_PROMPT", required=True)

    # and the optional environment variables
    text_generation_model_name = get_env_var_or_default("TEXT_GENERATION_MODEL", "gpt-4o")
    function_calling_model_name = get_env_var_or_default("FUNCTION_CALLING_MODEL", "gpt-3.5-turbo")
    fc_json_schema = get_env_var_or_default("FUNCTION_CALLING_VALIDATION_SCHEMA")
    fc_json_schema = fetch_content_from([fc_json_schema]) if fc_json_schema else None
    fc_json_schema = json.loads(fc_json_schema) if fc_json_schema else None
    github_output_file = get_env_var_or_default("GITHUB_OUTPUT")
    output_key = get_env_var_or_default("OUTPUT_KEY","text_generation")
    output_schema = get_env_var_or_default("OUTPUT_SCHEMA")
    output_schema = fetch_content_from([output_schema]) if output_schema else None
    output_schema = json.loads(output_schema) if output_schema else None
    bot_name = get_env_var_or_default("BOT_NAME")
    service_token = get_env_var_or_default("OPENAPI_SERVICE_TOKEN")
    user_prompt = get_env_var_or_default("USER_PROMPT")
    user_prompt = fetch_content_from([user_prompt]) if user_prompt else None
    service_response_subtree = get_env_var_or_default("SERVICE_RESPONSE_SUBTREE")
    user_instruction = (
        extract_custom_instruction(bot_name, user_prompt)
        if user_prompt and bot_name
        else None
    )

    if user_instruction and contains_skip_instruction(user_instruction):
        print("Exiting, user prompt contains the word 'skip'.")
        sys.exit(0)

    invoke_service_pipe = Pipeline()
    invoke_service_pipe.add_component(
        "openapi_service",
        OpenAPITool(
            generator_api=LLMProvider.OPENAI,
            generator_api_params={"api_key": Secret.from_env_var("OPENAI_API_KEY"),
                                  "model": function_calling_model_name,
                                  "api_base_url": os.environ.get("OPENAI_API_BASE_URL")},
        ),
    )
    invoke_service_pipe.add_component("final_prompt", OpenAPIServiceResponseTextGeneration())
    invoke_service_pipe.connect("openapi_service.service_response","final_prompt.openapi_service_response")
    service_response = invoke_service_pipe.run(
        data={
            "openapi_service": {
                "messages": [ChatMessage.from_user(function_calling_prompt)],
                "credentials": Secret.from_token(service_token),
                "spec": open_api_spec_url,
            },
            "final_prompt": {
                "subtree": service_response_subtree,
                "system_prompt": system_prompt_text,
                "user_prompt": user_instruction,
            },
        }
    )

    text_gen_messages = service_response["final_prompt"]["prompt_messages"]
    # we are now ready to generate the final text

    gen_text_pipeline = Pipeline()
    gen_text_pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            model=text_generation_model_name, generation_kwargs={"max_tokens": 2560}
        ),
    )
    gen_text_pipeline.add_component("post", LLMJSONFormatEnforcer())
    gen_text_pipeline.add_component("router", ConditionalRouter(gen_text_routes()))
    gen_text_pipeline.add_component("a1", OutputAdapter("{{json_schema | prepare_fc_params}}", Dict[str, Any], cf))
    gen_text_pipeline.add_component("a3", OutputAdapter("{{messages | change_role}}", List[ChatMessage], cf))

    gen_text_pipeline.add_component("a4", OutputAdapter("{{messages[0].content | json_loads}}", Dict[str, Any], cf))
    gen_text_pipeline.add_component(
        "a5",
        OutputAdapter(
            "{{json_payload[0]['function']['arguments'] | json_loads}}",
            Dict[str, Any],
            cf,
        ),
    )
    gen_text_pipeline.add_component("a6", OutputAdapter("{{messages[0].content | json_loads}}", Dict[str, Any], cf))
    gen_text_pipeline.add_component("json_gen_llm", OpenAIChatGenerator(model=function_calling_model_name))
    gen_text_pipeline.add_component("schema_validator", JsonSchemaValidator())
    gen_text_pipeline.add_component("mx_final_output", BranchJoiner(Dict[str, Any]))
    gen_text_pipeline.add_component("mx_for_json_gen_llm", BranchJoiner(List[ChatMessage]))
    gen_text_pipeline.add_component("final_output", FormattedOutputProcessor())
    gen_text_pipeline.connect("llm.replies", "post.messages")
    gen_text_pipeline.connect("post.messages", "router")
    gen_text_pipeline.connect("router.needs_function_calling", "mx_for_json_gen_llm")
    gen_text_pipeline.connect("router.no_need_for_function_calling", "a6.messages")
    gen_text_pipeline.connect("json_gen_llm.replies", "schema_validator.messages")
    gen_text_pipeline.connect("a5", "mx_final_output")
    gen_text_pipeline.connect("a6", "mx_final_output")
    gen_text_pipeline.connect("schema_validator.validation_error", "mx_for_json_gen_llm")
    gen_text_pipeline.connect("mx_for_json_gen_llm", "a3.messages")
    gen_text_pipeline.connect("a1", "json_gen_llm.generation_kwargs")
    gen_text_pipeline.connect("a3", "json_gen_llm.messages")
    gen_text_pipeline.connect("schema_validator.validated", "a4")
    gen_text_pipeline.connect("a4", "a5.json_payload")
    gen_text_pipeline.connect("mx_final_output", "final_output.json_object")

    output_sinks = (
        [FormattedOutputTarget(github_output_file, GITHUB_OUTPUT_TEMPLATE)]
        if github_output_file
        else []
    )
    # always output to stdout for debugging unless quiet mode is enabled
    if not get_env_var_or_default(env_var_name="QUIET", default_value=False):
        output_sinks.append(FormattedOutputTarget("stdout", STDOUT_OUTPUT_TEMPLATE))
    gen_text_pipeline.run(
        data={
            "messages": text_gen_messages,
            "has_output_schema": bool(output_schema),
            "output_key": output_key,
            "output_targets": output_sinks,
            "json_schema": output_schema,
        }
    )
