import os
import json
import importlib.util


def get_sample_data(app_name, data_name=None, sub_app=None):
    """Retrieve sample data for a given app, handling various data file types."""
    base_dir = os.path.dirname(__file__)  # Directory of the script
    path = os.path.join(base_dir, '..', 'sample_data', app_name)
    if sub_app:
        path = os.path.join(path, sub_app)

    # Determine file type and adjust path if needed
    if os.path.exists(path + '.py'):
        path += '.py'
        data = load_data_from_python(path, data_name)
    elif os.path.exists(path + '.json'):
        path += '.json'
        data = load_data_from_json(path, data_name)
    elif os.path.exists(path + '.txt'):
        path += '.txt'
        data = load_data_from_text(path)
    else:
        data = None  # No suitable file found

    return data


def load_data_from_python(path, var_name):
    """Import and return a specific variable from a Python file."""
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, var_name, None)


def load_data_from_json(path, key=None):
    """Load data from a JSON file, returning a specific key or entire data."""
    with open(path, 'r') as file:
        data = json.load(file)
    return data.get(key) if key else data


def load_data_from_text(path, encoding='utf-8'):
    """Read and return content from a text file."""
    "Change by Jatin: I added a default encoding."
    with open(path, 'r', encoding=encoding) as file:
        return file.read()


if __name__ == "__main__":
    # Example usage
    sample_data = get_sample_data(app_name="automation_matrix", data_name="markdown_content", sub_app="ama_ai_output_samples")
    print(sample_data)
