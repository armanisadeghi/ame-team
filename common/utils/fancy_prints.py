import inspect
import json
print(json.__file__)

import re
from decimal import Decimal
from uuid import UUID
import datetime


def vcprint(verbose=False, data=None, title="Unnamed Data", color="white", background="black", style="bold", indent=4):
    if not data:
        data = "No data provided."
    if verbose:
        pretty_print(data, title, color, background, style, indent)


def pretty_print(data, title="Unnamed Data", color="white", background="black", style="bold", indent=4):
    frame = inspect.currentframe()
    try:
        context = inspect.getouterframes(frame)
        if title == "Unnamed Data":
            name = title
            for var_name, var_val in context[1].frame.f_locals.items():
                if var_val is data:
                    name = var_name
                    break
        else:
            name = title

        if isinstance(data, str) and not data.strip().startswith(('{', '[')):
            if color:
                cool_print(text=f"\n{name}:\n{data}", color=color, background=background, style=style)
            else:
                print(f"\n{name}:\n{data}")
            return

        converted_data, old_type, new_type = convert_to_json_compatible(data)
        type_message = f" [{old_type} converted to {new_type}]" if old_type != new_type else ""
        json_string = json.dumps(converted_data, indent=indent)

        compact_json_string = re.sub(r'"\\"([^"]*)\\""', r'"\1"', json_string)
        compact_json_string = re.sub(r'\[\n\s+((?:\d+,?\s*)+)\n\s+\]', lambda m: '[' + m.group(1).replace('\n', '').replace(' ', '') + ']', compact_json_string)

        if color:
            cool_print(text=f"\n{name}:{type_message}\n{compact_json_string}", color=color, background=background, style=style)

    finally:
        del frame


def convert_to_json_compatible(data):
    """
    Recursively converts various data types into JSON-compatible formats.
    Returns a tuple of the converted data, the original data type, and the converted data type.
    """
    old_type = type(data).__name__
    new_type = old_type

    if isinstance(data, (str, int, float, bool, type(None), UUID)):
        return str(data), old_type, new_type
    elif isinstance(data, (list, tuple)):
        converted_list = [convert_to_json_compatible(item)[0] for item in data]
        new_type = "list" if isinstance(data, list) else "tuple"
        return converted_list, old_type, new_type
    elif isinstance(data, dict):
        converted_dict = {key: convert_to_json_compatible(value)[0] for key, value in data.items()}
        return converted_dict, old_type, "dict"
    elif isinstance(data, datetime.datetime):
        return data.isoformat(), old_type, "str"
    elif isinstance(data, Decimal):
        return float(data), old_type, "float"

    # Commented out for non-django setup
    # elif isinstance(data, Model):
    # return {field.name: convert_to_json_compatible(getattr(data, field.name))[0] for field in data._meta.fields}, old_type, "dict"
    # elif isinstance(data, QuerySet):
    # return [convert_to_json_compatible(obj)[0] for obj in data], old_type, "list[dict]"
    elif hasattr(data, 'dict'):
        return {key: convert_to_json_compatible(value)[0] for key, value in data.dict().items()}, old_type, "dict"
    else:
        try:
            return str(data), old_type, "str"
        except Exception:
            return "This data type is:", old_type, "which is not compatible with pretty print."


def print_link(path):
    from urllib.parse import urlparse
    import os

    if any(suffix in path.lower() for suffix in {'.com', '.org', '.net', '.io', '.us', '.gov'}):
        print(path)
        return

    if not isinstance(path, str):
        raise ValueError("The provided path must be a string.")

    parsed_path = urlparse(path)

    if parsed_path.scheme and parsed_path.netloc:
        print(path)

    else:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        url_compatible_path = path.replace("\\", "/")
        print("file:///{}".format(url_compatible_path))


def colorize(text, color=None, background=None, style=None):
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        # "white": "\033[1;37m",
        "black": "\033[30m",
    }
    backgrounds = {
        "red": "\033[41m",
        "green": "\033[42m",
        "yellow": "\033[43m",
        "blue": "\033[44m",
        "magenta": "\033[45m",
        "cyan": "\033[46m",
        "white": "\033[47m",
    }
    styles = {
        "bold": "\033[1m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "dim": "\033[2m",
    }
    reset = "\033[0m"
    color_code = colors.get(color, "")
    background_code = backgrounds.get(background, "")
    style_code = styles.get(style, "")

    return f"{color_code}{background_code}{style_code}{text}{reset}"


def cool_print(text, color, background=None, style=None):
    print(colorize(text, color, background, style))


def pretty_verbose(data, verbose=False, title=None):
    if verbose:
        pretty_print(data, title)


def cool_verbose(data, verbose=False, color="blue", background=None, style=None):
    if verbose:
        cool_print(text=data, color=color, background=background, style=style)


def vprint(verbose=False, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def print_red(text):
    print(colorize(text, "red"))


def print_green(text):
    print(colorize(text, "green"))


def print_yellow(text):
    print(colorize(text, "yellow"))


def print_blue(text):
    print(colorize(text, "blue"))


def print_bold(text):
    print(colorize(text, "bold"))


def print_bold_red(text):
    print(colorize(text, "red", style="bold"))


def print_bold_green(text):
    print(colorize(text, "green", style="bold"))


def print_bold_yellow(text):
    print(colorize(text, "yellow", style="bold"))


def print_bold_blue(text):
    print(colorize(text, "blue", style="bold"))


def print_bold_magenta(text):
    print(colorize(text, "magenta", style="bold"))


def print_bold_cyan(text):
    print(colorize(text, "cyan", style="bold"))


def print_bold_white(text):
    print(colorize(text, "white", style="bold"))


def print_underline_red(text):
    print(colorize(text, "red", style="underline"))


def print_underline_green(text):
    print(colorize(text, "green", style="underline"))


def print_underline_yellow(text):
    print(colorize(text, "yellow", style="underline"))


def print_underline_blue(text):
    print(colorize(text, "blue", style="underline"))


def print_blink(text):
    print(colorize(text, "red", style="blink"))


def print_blink_red(text):
    print(colorize(text, "red", style="blink"))


def print_blink_green(text):
    print(colorize(text, "green", style="blink"))


def print_blink_yellow(text):
    print(colorize(text, "yellow", style="blink"))


def print_blink_blue(text):
    print(colorize(text, "blue", style="blink"))


def print_dim(text):
    print(colorize(text, "red", style="dim"))


def print_dim_red(text):
    print(colorize(text, "red", style="dim"))


def print_dim_green(text):
    print(colorize(text, "green", style="dim"))


def print_dim_yellow(text):
    print(colorize(text, "yellow", style="dim"))


def print_dim_blue(text):
    print(colorize(text, "blue", style="dim"))


def print_background_red(text):
    print(colorize(text, "white", background="red"))


def print_background_green(text):
    print(colorize(text, "white", background="green"))


def print_background_yellow(text):
    print(colorize(text, "white", background="yellow"))


def print_background_blue(text):
    print(colorize(text, "white", background="blue"))
