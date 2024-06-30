import copy
import functools
import inspect
import json
import re
from typing import get_type_hints

from classifier import TextAnalyzer
from identifiers import LineIdentifier
from metrics import Metric
from operations import LineOperations, FindOperations
from utils import generate_random_key


def find_operation_requires_identifiers(find_obj, action: str) -> bool:
    # What this function does is just get the type hints, and if anything requires a list of LineIdentifiers, it returns True
    if not hasattr(find_obj, action):
        return False

    method = getattr(find_obj, action)

    if not inspect.isfunction(method) and not inspect.ismethod(method):
        return False

    type_hints = get_type_hints(method)

    for param, hint in type_hints.items():
        if hint == list[LineIdentifier]:
            return True
    return False


class Document:

    def __init__(self, lines=None):
        self.lines = lines

    def get_text(self):
        return "\n".join([x.get('line') for x in self.lines])

    def from_lines(self, lines: list[str]):
        doc = '\n'.join(lines)
        classified_lines = TextAnalyzer(text=doc).get_analysis().get('lines')
        self.lines = classified_lines


with open('metric data/all_metrics.json') as f:
    metric_list = json.loads(f.read())


def save_state_by_array(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Call the `get_current_state` method to get the dictionary
        state_dict = self.get_state()

        # Save the dictionary to a JSON file
        filename = 'output/state_store_array.json'
        with open(filename, 'w') as f:
            json.dump(state_dict, f, indent=4)

        return result

    return wrapper


class Node:
    def __init__(self, line_number, content):
        self.line_number = line_number
        self.content = content
        self.prev = None
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.line_map = {}

    def append(self, line_number, content):
        new_node = Node(line_number, content)
        self.line_map[line_number] = new_node
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def insert_before(self, line_number, content):
        if line_number not in self.line_map:
            return
        existing_node = self.line_map[line_number]
        new_node = Node(f"new_{generate_random_key()}", content)
        new_node.next = existing_node
        new_node.prev = existing_node.prev
        if existing_node.prev:
            existing_node.prev.next = new_node
        existing_node.prev = new_node
        if existing_node == self.head:
            self.head = new_node

    def insert_after(self, line_number, content):
        if line_number not in self.line_map:
            return
        existing_node = self.line_map[line_number]
        new_node = Node(f"new_{generate_random_key()}", content)
        new_node.prev = existing_node
        new_node.next = existing_node.next
        if existing_node.next:
            existing_node.next.prev = new_node
        existing_node.next = new_node
        if existing_node == self.tail:
            self.tail = new_node

    def edit_line(self, line_number, content):
        if line_number not in self.line_map:
            return
        node = self.line_map[line_number]
        node.content = content

    def get_lines(self):
        return_obj = {}
        current = self.head
        while current:
            return_obj[current.line_number] = current.content
            current = current.next

        return return_obj


class TextProcessor:
    def __init__(self, document_lines):

        self.lines = self.get_lines(document_lines)
        self.metric_list = metric_list
        self.document_lines = document_lines
        self.added_metrics = []
        self.metric_results = []
        self.identifiers = []
        self.searches = []
        self.steps = []
        self.updated_document_lines = []

    def get_lines(self, doc_line_list):
        doc_ = Document()
        doc_.from_lines(doc_line_list)
        return doc_.lines

    def get_saved_metrics(self):
        saved_metrics = []
        for results in self.metric_results:
            metric_index = results.get('added_metric')
            passed_lines = results.get('lines')
            metric_dict = self.eval_incoming_metric(self.added_metrics[metric_index])

            metric_obj = Metric()
            metric_obj.from_dict(metric_dict)

            saved_metrics.append({
                "metric": metric_obj,
                "passed_lines": passed_lines
            })
        return saved_metrics

    def eval_incoming_metric(self, metric_structure):
        # Expects a structure like :
        # {
        #     "metric_index" : 0,
        #     "parameters": {
        #         "equals": "Somethings",
        #         "case_sensitive": True
        #     }
        # }
        metric_index = metric_structure["metric_index"]
        metric_name = self.metric_list[metric_index]
        parameters = metric_structure["parameters"]

        metric_dict = {
            "metric": metric_name,
            **parameters
        }

        # Returns something like this
        # {"metric": 'starts_with', 'case_sensitive': True, 'equals': "Somethings"}
        return metric_dict

    @save_state_by_array
    def handle_incoming_metric(self, metric_structure):
        dict_metric = self.eval_incoming_metric(metric_structure)

        self.added_metrics.append(metric_structure)

        total_added_metrics = len(self.added_metrics)

        metric_store_index = total_added_metrics - 1

        # Metric evaluation
        metric_obj = Metric()
        metric_obj.from_dict(dict_metric)

        passed_lines = []
        for line in self.lines:
            line_passes_test = metric_obj.evaluate(line, self.lines)

            if line_passes_test:
                passed_lines.append(line.get('line_number'))

        self.metric_results.append({
            "added_metric": metric_store_index,
            "lines": passed_lines
        })

    def eval_incoming_identifier(self, identifier_structure):
        for k, v in identifier_structure.items():
            metric_indices = v.get('metrics')
            v['metrics'] = []

            for m in metric_indices:
                v['metrics'].append(self.eval_incoming_metric(self.added_metrics[m]))

        return identifier_structure

    @save_state_by_array
    def handle_incoming_identifier(self, identifier_structure):
        self.identifiers.append(identifier_structure)

    @save_state_by_array
    def process_search(self, search_structure):
        todo_action = search_structure.get('action')
        kwargs = search_structure.get('kwargs')
        name = search_structure.get('name', None)

        find_ops = FindOperations(self.lines, self.get_saved_metrics())

        if find_operation_requires_identifiers(find_ops, todo_action):
            # Create a deep copy of kwargs to avoid modifying the original
            kwargs_copy = copy.deepcopy(kwargs)
            identifier_indices = kwargs_copy.get('identifiers', [])

            identifiers = [
                self.eval_incoming_identifier(copy.deepcopy(self.identifiers[i]))
                for i in identifier_indices
            ]

            identifiers_obj_list = []

            for idf_dict in identifiers:
                idf_obj = LineIdentifier()
                idf_obj.from_dict(idf_dict)
                identifiers_obj_list.append(idf_obj)

            kwargs_copy['identifiers'] = identifiers_obj_list

            output = getattr(find_ops, todo_action)(**kwargs_copy)
        else:
            output = getattr(find_ops, todo_action)(**kwargs)

        self.searches.append({
            "name": name,
            "action": todo_action,
            "kwargs": search_structure.get('kwargs'),
            "output": output  # Output generally will be list[int] or list[list[int, int]]
        })

    @save_state_by_array
    def process_steps(self, edit_structure):
        action = edit_structure.get('action')
        kwargs = edit_structure.get('kwargs')

        # This uses the LineOperations Class
        # The LineOperations class just returns a bunch of add, delete, edit set of instructions
        # We are not using the class now, for now just storing the steps
        self.steps.append({
            "action": action,
            "kwargs": kwargs
        })

    def process_by_edits(self, edits: list):

        doc_ = LinkedList()

        for idx, line in enumerate(self.document_lines, start=1):
            doc_.append(idx, line)

        delete_lines = []

        for edit in edits:
            keys = list(edit.keys())  # Get the type of edit and the value
            edit_type = keys[0]
            edit_lines = edit[keys[0]]

            # Incase new lines need to be added, we want to add line with random keys either above or down in order of the edit,
            if edit_type == "new_lines":
                for line in edit_lines:
                    if line['position'] == "before":
                        doc_.insert_before(line['line_number'], line['content'])
                    elif line['position'] == "after":
                        doc_.insert_after(line['line_number'], line['content'])

            # Incase lines need to be deleted
            elif edit_type == "delete_lines":
                delete_lines.extend(edit_lines)

            # Incase lines need to be edited
            elif edit_type == "edit_lines":
                for line_ in edit_lines:
                    line_number = line_.get('line_number')
                    line_content = self.document_lines[line_number - 1]
                    content = line_.get('content')
                    add_content = line_.get('add_content')
                    replace_with = line_.get('with')
                    replace_pattern = line_.get('replace_pattern')
                    replace = line_.get('replace')

                    if replace and replace_with:
                        if replace in line_content:
                            line_content = line_content.replace(replace, replace_with)
                            doc_.edit_line(line_number, line_content)

                    elif replace_pattern and replace_with:
                        line_content = re.sub(replace_pattern, replace_with, line_content)
                        doc_.edit_line(line_number, line_content)

                    elif content:
                        doc_.edit_line(line_number, content)

                    elif add_content:
                        doc_.edit_line(line_number, f"{line_content}{add_content}")

        return [val for num, val in doc_.get_lines().items() if num not in delete_lines]

    def get_snapshot(self, step_index: int):
        steps_to_process = self.steps[0: step_index + 1]
        edits = []

        for step in steps_to_process:
            kwargs = step.get('kwargs')
            lines_ops = LineOperations(self.lines)

            if getattr(lines_ops, step.get('action')):
                output = getattr(lines_ops, step.get('action'))(**kwargs)
                print("Output from edit maker", output)
                if isinstance(output, list):
                    edits.extend(output)

        output = self.process_by_edits(edits)
        with open(f'output/output_{step_index}.json', 'w') as f:
            f.write(json.dumps(output))
        return output

    def process(self):
        processed_lines = self.get_snapshot(len(self.steps))
        return processed_lines

    def get_state(self):
        return {
            "added_metrics": self.added_metrics,
            "metric_results": self.metric_results,
            "identifiers": self.identifiers,
            "steps": self.steps,
            "searches": self.searches,

        }

    def load_from_state(self, config):
        self.steps = config.get('steps')
        self.added_metrics = config.get('added_metrics')
        self.metric_results = config.get('metric_results')
        self.identifiers = config.get('identifiers')
        self.searches = config.get('searches')


if __name__ == "__main__":
    with open('text samples/ama_raw_full.txt', encoding='utf-8') as f:
        doc = f.read()

    doc_lines = doc.split('\n')

    processor = TextProcessor(doc_lines)

    # M0
    processor.handle_incoming_metric({"metric_index": 0, "parameters": {"case_sensitive": True, "equals": "Chapter"}})
    # M1
    processor.handle_incoming_metric({"metric_index": 9, "parameters": {"equals": True}})

    # Char Count # M2
    processor.handle_incoming_metric({"metric_index": 27, "parameters": {"less_than": 1}})

    # Reference line # M3
    processor.handle_incoming_metric({"metric_index": 0, "parameters": {"equals": "references", "case_sensitive": False}})

    # Reference line # M4
    processor.handle_incoming_metric(
        {"metric_index": 1, "parameters": {"equals": "references", "case_sensitive": False}})

    # CVC line # M5
    processor.handle_incoming_metric({"metric_index": 32, "parameters": {"equals": True}})


    # Chapter line identifier, # I0
    processor.handle_incoming_identifier({
        "0": {
            "type": "AND",
            "metrics": [0, 1]
        }
    })

    # Empty Line identifier # I1
    processor.handle_incoming_identifier({
        "0": {
            "type": "AND",
            "metrics": [2]
        }
    })

    # Reference Line Identifier # I2
    processor.handle_incoming_identifier({
        "0": {
            "type": "AND",
            "metrics": [3,4]
        }
    })

    # CVC line idf # I3
    processor.handle_incoming_identifier({
        "0":{
            "type": "AND",
            "metrics": [1,5]
        }
    })

    # Find chapter lines , # S0
    processor.process_search({
        "action": "find_lines_by_identifier",
        "kwargs": {
            "identifiers": [0]
        }
    })

    # Find Consecutive Empty Lines, #S1
    processor.process_search({
        "action": "find_blocks_by_identifier",
        "kwargs": {
            "identifiers": [1],
            "config" : [{'optional': False, 'min': 3}],
            "size": 3
        }
    })

    # S2, find Refs
    processor.process_search(
        {
            "action": 'find_blocks_between_identifiers',
            'kwargs': {
                "identifiers": [2, 0],
                "max_lookup": 150,
                "max_size": 50
            }
        }
    )

    # CVC search # S3
    processor.process_search({
        "action": "find_lines_by_identifier",
        "kwargs": {
            "identifiers" : [3]
        }
    })

    # Step 1 Limiting Consective Empty Lines
    processor.process_steps(
        {"action": "delete_line",
         "kwargs": {"line_numbers": [y for x in processor.searches[1].get('output') for y in range(x[0], x[1])]}})

    # Step 2 : Adding Chapter Line Markers
    processor.process_steps(
        {"action": "add_line_before", "kwargs": {"line_numbers": processor.searches[0].get('output'), "text": "-- start chapter --"}})

    processor.process_steps(
        {"action": "add_line_after",
         "kwargs": {"line_numbers": processor.searches[0].get('output'), "text": "-- end chapter --"}})

    # Step 3: Getting out references sections and removing them #S2


    processor.process_steps({
        "action": "delete_line",
        "kwargs" : {
            "line_numbers": [y for x in processor.searches[2].get('output') for y in range(x[0], x[1])],
        }
    })


    # Step 4 Clearing out CVCs

    processor.process_steps({
        'action': 'delete_line',
        'kwargs': {
            "line_numbers"  : processor.searches[3].get('output')
        }
    })

    processor.process()