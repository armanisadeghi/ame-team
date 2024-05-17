import asyncio
from common import vcprint, get_sample_data, pretty_print

verbose = True


# Helper function to extract brokers from the processed data and simulate what the workflow would do.
def extract_brokers(processed_content):
    def recurse_brokers(values):
        if isinstance(values, dict):
            for key, value in values.items():
                if key == 'brokers':
                    for broker_list in value:
                        for broker_name, broker_data in broker_list.items():
                            if broker_name not in brokers:
                                brokers[broker_name] = []
                            if isinstance(broker_data, list):
                                brokers[broker_name].extend(broker_data)
                            else:
                                brokers[broker_name].append(broker_data)
                else:
                    recurse_brokers(value)
        elif isinstance(values, list):
            for item in values:
                recurse_brokers(item)

    brokers = {}
    if 'processed_values' in processed_content:
        for processor_name, processor_data in processed_content['processed_values'].items():
            recurse_brokers(processor_data)

    brokers_and_values = brokers
    return brokers_and_values


# Associates brokers with args so they can be easily and directly used for a function call.
def filter_and_rename_brokers(brokers, broker_arg_mappings):
    # Create a dictionary to easily look up arg names by broker name
    arg_lookup = {list(b.keys())[0]: list(b.values())[0] for b in broker_arg_mappings}

    # Filter and rename the brokers based on the provided mappings
    filtered_brokers = {}
    for broker_name, broker_data in brokers.items():
        if broker_name in arg_lookup:
            arg_name = arg_lookup[broker_name]
            filtered_brokers[arg_name] = broker_data

    return filtered_brokers


# This represents the actual function you are calling with the extracted brokers and values.
def sample_actual_function_process_blog_content(topic, concept):
    print("\n\n")
    print("-" * 50)
    print("Topic:", topic)
    print("Concept:", concept)
    print("-" * 50)

    some_function_result = "Function call successful!"

    return some_function_result


# This can unpack the args and make the call.
def simulate_actual_function_call(args_structure):
    adjusted_brokers = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in args_structure.items()}

    # Now make the function call using the ** syntax to unpack keyword arguments
    result = sample_actual_function_process_blog_content(**adjusted_brokers)
    return result


def simulate_workflow_after_processing(processor_results):

    # Right now, this is missing a step where you would get the

    brokers_and_values = extract_brokers(processor_results)
    vcprint(verbose=True, data=brokers_and_values, title="Brokers & Values", color='green', style='bold')

    broker_arg_mappings = [  # This is a sample of what you might try to extract from the processed content to have args associated with values. Match the broker name to the arg name you need.
        {
            "BLOG_IDEA_1": "topic"
        },
        {
            "SAMPLE_DATA_1001": "concept"
        }
    ]

    args_structure = filter_and_rename_brokers(brokers_and_values, broker_arg_mappings)
    vcprint(verbose=True, data=args_structure, title="Final Args for Function Call", color='blue', style='bold')

    wf_simulation_results = simulate_actual_function_call(args_structure)
    vcprint(verbose=True, data=wf_simulation_results, title="Final Result", color='cyan', style='bold')

    return wf_simulation_results
