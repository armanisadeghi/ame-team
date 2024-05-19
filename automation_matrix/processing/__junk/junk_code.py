
# TODO: Jatin, I need you to fine out if this is actually an extractor that is misplaced or if it's some sort of duplicate code. Please put it in the right place.
# Ask me if you can't figure it out. Basically, if it's necessary for returning the proper final structure, then it's an extractor and needs to be properly placed.
async def access_data_by_reference(reference, data_structure):
    """
    Access a nested dictionary entry based on a reference dictionary and return it as specified (text or dict).

    :param reference: A dictionary with 'data', 'key', and 'output' where 'data' is the key to the main dictionary,
                      'key' is the index to access within the nested dictionary (None for entire structure), and
                      'output' specifies the return format ('text' or 'dict').
    :param data_structure: The main data structure containing nested dictionaries.
    :return: The selected entry in the specified format, or a message if not found.

        # Sample entry to get the 3rd key as text
        reference_text_3rd = {"data": "nested_structure", "key": 1, "output": "text"}

        # Sample entry to get the 2nd key as a dict
        reference_dict_2nd = {"data": "nested_structure", "key": 1, "output": "dict"}

        # Sample entry to get the entire nested structure as text
        reference_entire_text = {"data": "nested_structure", "key": None, "output": "text"}

    """

    nested_key = reference.get('key_identifier')
    index = reference.get('key_index')
    output_format = reference.get('output_type', 'dict')  # Default to 'dict' if not specified
    pretty_print(reference)
    if not index and index != 0:
        nested_dict = data_structure.get(nested_key, {})
        if output_format == 'text':
            result = '\n\n'.join(f'{key}:\n{'\n'.join(value)}' for key, value in nested_dict.items())
        else:
            result = nested_dict
    else:
        index -= 1  # Adjust for 0-based indexing
        if nested_key in data_structure:
            nested_dict = data_structure[nested_key]
            keys = list(nested_dict.keys())
            if 0 <= index < len(keys):
                selected_key = keys[index]
                content = nested_dict[selected_key]
                if output_format == 'text':
                    result = f"{selected_key}:\n{'\n'.join(content)}"
                else:
                    result = {
                        selected_key: content
                    }
            else:
                result = "The specified entry was not found."
        else:
            result = "The specified entry was not found."

    return result


# TODO: This is another one that I'm not sure actually belongs here or if it's just leftover test code. If it isn't real code that we need for processing or extracting, then it needs to be removed.
async def handle_OpenAIWrapperResponse(result):
    core_variable_name = result.get('variable_name', '')
    processed_values = result.get('processed_values', {})
    p_index = 0
    print(f"Processing {core_variable_name}...")
    if result.get('processing'):
        print("Processing is still in progress.")
        for processor, processor_data in processed_values.items():
            p_index += 1
            e_index = 0
            method_name = f"handle_{processor}"
            processor_value = processor_data.get('value', {})
            if 'extraction' in processor_data:
                for extraction_map in processor_data['extraction']:  # Adjusted to iterate over a list
                    print(f"-------------Processing {core_variable_name} with {method_name}...")
                    e_index += 1
                    variable_name = f"{p_index}_{e_index}_{core_variable_name}"
                    print(f"Processing {variable_name}")

                    try:
                        extraction_result = await access_data_by_reference(extraction_map, processor_value)  # Changed variable name from 'result' to 'extraction_result' to avoid overshadowing
                        pretty_print(extraction_result)
                        print(f"==================================================")
                    except Exception as ex:
                        # Log the error and continue with the next extraction_map
                        print(f"Error processing {variable_name} with {method_name}: {ex}")
