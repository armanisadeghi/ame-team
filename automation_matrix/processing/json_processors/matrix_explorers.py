import json
import re
import datetime
from decimal import Decimal
import inspect
import asyncio


class JSONExplorer:
    """
    A class to explore and search JSON data, with support for both synchronous and asynchronous operation.
    This class is capable of initializing with either JSON data or a file path. It includes functionalities
    for data type conversion, pretty printing, and various data manipulation methods.
    """

    def __init__(self, data):
        """
        Initializes the JSONExplorer with either a JSON file path or JSON data.
        Automatically converts non-JSON compatible data types using convert_to_json_compatible.
        """
        if isinstance(data, str):
            try:
                with open(data, 'r') as file:
                    self.data = json.load(file)
                self.file_path = data  # Store the file path for saving changes later
            except FileNotFoundError:
                raise Exception(f"File not found: {data}")
        else:
            self.data, _, _ = self.convert_to_json_compatible(data)
            self.file_path = None

    def get_config_value(self, key_path, default=None):
        """
        Retrieves a configuration value based on a key path.
        """
        keys = key_path.split('.')
        data = self.data
        for key in keys:
            if key in data:
                data = data[key]
            else:
                return default
        return data

    def list_keys(self, data=None):
        """
        Lists all keys in the JSON data at the current level.
        """
        if data is None:
            data = self.data
        if isinstance(data, dict):
            return list(data.keys())
        return []

    def recursive_traversal(self, data=None, parent_key=''):
        """
        Recursively traverses the JSON data and prints all keys.
        """
        if data is None:
            data = self.data

        if isinstance(data, dict):
            for key, value in data.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                print(current_key)
                self.recursive_traversal(value, current_key)
        elif isinstance(data, list):
            for item in data:
                self.recursive_traversal(item, parent_key)

    def find_by_partial_key(self, partial_key, data=None):
        """
        Finds keys that include the partial key name.
        """
        if data is None:
            data = self.data

        found = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if partial_key in key:
                    found[key] = value
                found.update(self.find_by_partial_key(partial_key, value))
        elif isinstance(data, list):
            for item in data:
                found.update(self.find_by_partial_key(partial_key, item))
        return found

    def find_by_regex(self, pattern, data=None):
        """
        Finds keys that match a regular expression pattern.
        """
        if data is None:
            data = self.data

        found = {}
        regex = re.compile(pattern)
        if isinstance(data, dict):
            for key, value in data.items():
                if regex.search(key):
                    found[key] = value
                found.update(self.find_by_regex(pattern, value))
        elif isinstance(data, list):
            for item in data:
                found.update(self.find_by_regex(pattern, item))
        return found

    def get_nested_value(self, nested_key, data=None):
        """
        Retrieves the value for a nested key if it exists anywhere in the JSON data.
        """
        if data is None:
            data = self.data

        if isinstance(data, dict):
            for key, value in data.items():
                if key == nested_key:
                    return value
                elif isinstance(value, (dict, list)):
                    result = self.get_nested_value(nested_key, value)
                    if result is not None:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = self.get_nested_value(nested_key, item)
                if result is not None:
                    return result
        return None

    def update_value(self, key_path, new_value):
        """
        Updates the value for a given key path in the JSON data.
        """
        keys = key_path.split('.')
        data = self.data
        for key in keys[:-1]:
            data = data.setdefault(key, {})
        data[keys[-1]], _, _ = self.convert_to_json_compatible(new_value)
        if self.file_path:
            self.save_changes()

    def save_changes(self):
        """
        Saves the current state of the JSON data back to the file.
        """
        if self.file_path:
            with open(self.file_path, 'w') as file:
                json.dump(self.data, file, indent=4)
        else:
            raise Exception("No file path provided. Data cannot be saved to a file.")

    def pretty_print_json(self, data=None, indent=4):
        """
        Prints the JSON data or a subset of it in a formatted, human-readable way.
        """
        if data is None:
            data = self.data
        print(f"\nPretty Printed JSON:\n{json.dumps(data, indent=indent)}\n")

    @staticmethod
    def convert_to_json_compatible(data):
        """
        Recursively converts various data types into JSON-compatible formats.
        """
        old_type = type(data).__name__
        new_type = old_type

        if isinstance(data, (str, int, float, bool, type(None))):
            return data, old_type, new_type
        elif isinstance(data, (list, tuple)):
            converted_list = [JSONExplorer.convert_to_json_compatible(item)[0] for item in data]
            new_type = "list" if isinstance(data, list) else "tuple"
            return converted_list, old_type, new_type
        elif isinstance(data, dict):
            converted_dict = {key: JSONExplorer.convert_to_json_compatible(value)[0] for key, value in data.items()}
            return converted_dict, old_type, "dict"
        elif isinstance(data, datetime.datetime):
            return data.isoformat(), old_type, "str"
        elif isinstance(data, Decimal):
            return float(data), old_type, "float"
        elif hasattr(data, '__dict__'):
            converted_obj = {key: JSONExplorer.convert_to_json_compatible(value)[0] for key, value in data.__dict__.items()}
            return converted_obj, old_type, "dict"
        else:
            try:
                return str(data), old_type, "str"
            except Exception:
                return "This data type is not compatible with pretty print.", old_type, "incompatible"

    def pretty_print(self, data=None, indent=4, called_directly=False):
        """
        Prints the JSON data or a subset of it in a formatted, human-readable way, including the variable name
        it was called with. This method has been enhanced to match the original standalone pretty_print_data
        functionality.
        """
        if data is None:
            data = self.data

        # Check if data is a string and print it directly
        if isinstance(data, str):
            print(data)
            return

        # Attempt to find the variable name if this method was called directly
        name = "data"
        if called_directly:
            frame = inspect.currentframe().f_back
            try:
                context = inspect.getouterframes(frame)
                for var_name, var_val in context[1].frame.f_locals.items():
                    if var_val is data:
                        name = var_name
                        break
            finally:
                del frame

        try:
            # Attempt to convert data to JSON-compatible format and print it
            converted_data, old_type, new_type = self.convert_to_json_compatible(data)
            type_message = f" [{old_type} converted to {new_type}]" if old_type != new_type else ""
            print(f"\nPretty {name}:{type_message}\n{json.dumps(converted_data, indent=indent)}\n")
        except Exception as e:
            # Print an error message instead of raising an exception
            print(f"Error while trying to pretty print {name}: {e}")

    async def async_wrapper(self, method, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, getattr(self, method), *args, **kwargs)

    async def get_config_value_async(self, key_path, default=None):
        return await self.async_wrapper('get_config_value', key_path, default)


def pretty_print_data(data, indent=4):
    """
    Global function to pretty print any data, attempting to include the variable name used in the call.
    """
    explorer = JSONExplorer(data)
    explorer.pretty_print(data, indent, called_directly=True)

async def pretty_print_data_async(data, indent=4):
    """
    Asynchronous global function to pretty print any data, including the variable name if possible.
    """
    explorer = JSONExplorer(data)
    await explorer.async_wrapper('pretty_print', data, indent, True)


# Direct entry point for getting a configuration value
def get_config_value(data, key_path, default=None):
    explorer = JSONExplorer(data)
    return explorer.get_config_value(key_path, default)


# Asynchronous version
async def get_config_value_async(data, key_path, default=None):
    explorer = JSONExplorer(data)
    return await explorer.async_wrapper('get_config_value', key_path, default)


# Direct entry point for listing keys
def list_keys(data):
    explorer = JSONExplorer(data)
    return explorer.list_keys()


# Asynchronous version
async def list_keys_async(data):
    explorer = JSONExplorer(data)
    return await explorer.async_wrapper('list_keys')


def recursive_traversal(data, parent_key=''):
    explorer = JSONExplorer(data)
    explorer.recursive_traversal(data, parent_key)


async def recursive_traversal_async(data, parent_key=''):
    explorer = JSONExplorer(data)
    await explorer.async_wrapper('recursive_traversal', data, parent_key)


# Direct entry point for finding by partial key
def find_by_partial_key(data, partial_key):
    explorer = JSONExplorer(data)
    return explorer.find_by_partial_key(partial_key)


# Asynchronous version
async def find_by_partial_key_async(data, partial_key):
    explorer = JSONExplorer(data)
    return await explorer.async_wrapper('find_by_partial_key', partial_key)


# Direct entry point for finding by regex
def find_by_regex(data, pattern):
    explorer = JSONExplorer(data)
    return explorer.find_by_regex(pattern)


# Asynchronous version
async def find_by_regex_async(data, pattern):
    explorer = JSONExplorer(data)
    return await explorer.async_wrapper('find_by_regex', pattern)


# Direct entry point for getting a nested value
def get_nested_value(data, nested_key):
    explorer = JSONExplorer(data)
    return explorer.get_nested_value(nested_key)


# Asynchronous version
async def get_nested_value_async(data, nested_key):
    explorer = JSONExplorer(data)
    return await explorer.async_wrapper('get_nested_value', nested_key)


# Direct entry point for updating a value
def update_value(data, key_path, new_value):
    explorer = JSONExplorer(data)
    explorer.update_value(key_path, new_value)


# Asynchronous version
async def update_value_async(data, key_path, new_value):
    explorer = JSONExplorer(data)
    await explorer.async_wrapper('update_value', key_path, new_value)


def save_changes(data, file_path=None):
    explorer = JSONExplorer(data)
    if file_path:
        explorer.file_path = file_path
    explorer.save_changes()


async def save_changes_async(data, file_path=None):
    explorer = JSONExplorer(data)
    if file_path:
        explorer.file_path = file_path
    await explorer.async_wrapper('save_changes')

