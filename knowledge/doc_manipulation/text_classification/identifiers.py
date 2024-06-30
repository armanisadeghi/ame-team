from collections import defaultdict
from conditions import Condition
from metrics import Metric
from typing import Any, Dict, List


class LineIdentifier:
    """
    Example of a metric list:

        metrics = [
            ('metric1', {'option1': 'value1'}),
            ('metric2', {'option2': 'value2'}),
            ('metric3', {'option3': 'value3'}),
        ]

    **"AND": "Used for ensuring all conditions are true",**

    **"NOT": "Used for ensuring the condition is false",**

    **"OR": "Used for ensuring at least one condition is true",**

    **"XOR": "Used for ensuring exactly one condition is true",**

    **"NAND": "Used for ensuring not all conditions are true",**

    **"NOR": "Used for ensuring all conditions are false",**

    **"XNOR": "Used for ensuring conditions are either all true or all false",**

    **"IMPLICATION": "Used for ensuring if the first condition is true, the second must be true",**

    **"BICONDITIONAL": "Used for ensuring conditions are both true or both false"**
    """

    def __init__(self):
        self.identifier = defaultdict()
        self.idx = None
        self.groups_meanings = {
            "AND": "Used for ensuring all conditions are true",
            "NOT": "Used for ensuring the condition is false",
            "OR": "Used for ensuring at least one condition is true",
            "XOR": "Used for ensuring exactly one condition is true",
            "NAND": "Used for ensuring not all conditions are true",
            "NOR": "Used for ensuring all conditions are false",
            "XNOR": "Used for ensuring conditions are either all true or all false",
            "IMPLICATION": "Used for ensuring if the first condition is true, the second must be true",
            "BICONDITIONAL": "Used for ensuring conditions are both true or both false"
        }
        self.conditions: list[Condition] = []

    def add_condition(self, condition: Condition):
        self.conditions.append(condition)

    def evaluate(self, line_obj, lines, use_saved_metrics=False, saved_metrics: List[Dict[str, Any]] = None):

        if use_saved_metrics and saved_metrics is not None:
            for condition in self.conditions:
                if not condition.cached_evaluation(line_obj, saved_metrics):
                    return False
            return True
        else:

            for condition in self.conditions:
                if not condition.evaluate(line_obj, lines):
                    return False
            return True

    def _add_condition(self, conditional: str, metrics: list):
        conditional_type = conditional
        metrics = [Metric(metric_name, metric_params) for metric_name, metric_params in metrics]

        condition_obj = Condition(conditional_type, metrics)

        self.conditions.append(condition_obj)

    def ADD_AND(self, metrics):
        self._add_condition("AND", metrics=metrics)

    def ADD_NOT(self, metrics):
        self._add_condition("NOT", metrics=metrics)

    def ADD_OR(self, metrics):
        self._add_condition("OR", metrics=metrics)

    def ADD_XOR(self, metrics):
        self._add_condition("XOR", metrics=metrics)

    def ADD_NAND(self, metrics):
        self._add_condition("NAND", metrics=metrics)

    def ADD_NOR(self, metrics):
        self._add_condition("NOR", metrics=metrics)

    def ADD_XNOR(self, metrics):
        self._add_condition("XNOR", metrics=metrics)

    def ADD_IMPLICATION(self, metrics):
        if len(metrics) < 2:
            return
        self._add_condition("IMPLICATION", metrics=metrics[:1])

    def ADD_BICONDITIONAL(self, metrics):
        if len(metrics) < 2:
            return
        self._add_condition("BICONDITIONAL", metrics=metrics[:1])

    def reset(self):
        self.conditions = []

    def validate_structure_from_dict(self, identifier):

        if not isinstance(identifier, dict):
            return False

        if not all([isinstance(x, str) and x.isdigit() for x in identifier.keys()]):
            return False

        if not all([isinstance(x, dict) for x in identifier.values()]):
            return False

        if not all([x.get('type') in self.groups_meanings.keys() for x in identifier.values()]):
            return False

        if not all([isinstance(x.get('metrics'), list) for x in identifier.values()]):
            return False

        return True

    def from_dict(self, identifier: dict):
        # Utility function to load identifier from db (possibly)

        # Validating structure
        if not self.validate_structure_from_dict(identifier):
            return False

        for _, idf in identifier.items():
            metric_type = idf.get('type')
            metrics = idf.get('metrics')

            formatted_metrics = []
            for metric in metrics:
                formatted_metrics.append((metric.get('metric'), metric))

            self._add_condition(metric_type, formatted_metrics)

        return True

    def to_dict(self):
        structure = {}
        for idx, condition in enumerate(self.conditions):
            _type = condition.condition_type
            _metrics = condition.metrics
            _str_idx = str(idx)
            structure[_str_idx] = {
                "type": _type,
                "metrics": []
            }
            for _metric in _metrics:
                name, params = _metric.name, _metric.params

                params['metric'] = name

                structure[_str_idx]['metrics'].append(params)

        return structure