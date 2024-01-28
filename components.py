import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import time
from typing import List, Dict, Any, Union, TextIO
from typing import Optional

from haystack import component
from haystack.core.component.types import Variadic
from haystack.dataclasses import ChatMessage
from jinja2 import Template
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from dataclasses import dataclass

from util import ensure_json_objects, get_env_var_or_default, fetch_content_from


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
