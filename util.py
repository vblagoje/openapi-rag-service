import json
import logging
import os
import re
from typing import List, Dict, Any, Tuple
from typing import Optional
from urllib.parse import urlparse

import requests

GITHUB_OUTPUT_TEMPLATE = """
{{output_name}}<<{{delimiter()}}
{{render(output_value)}}
{{delimiter()}}
"""

STDOUT_OUTPUT_TEMPLATE = """
{{output_name}}:
{{render(output_value)}}
"""


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
