from metrics import Metric
from typing import List, Any, Dict

class Condition:
    def __init__(self,
                 condition_type: str,
                 metrics: list[Metric],
                 ):

        self.condition_type = condition_type
        self.metrics = metrics

    def cached_evaluation(self,
                          line_obj,
                          stored_evaluations: List[Dict[str, Any]],
                          ):
        """
        This is for cached evaluations only, cached means using results from stored metric evaluations
        :param line_obj: Line object
        :param stored_evaluations: The stored evaluations
        :return:

        example of stored evaluations
        stored_evaluations = {"metric1": [1,3,5], "metric2": [1,8,9], "metric3": [1,12, 13]}
        metrics =  {metric1 : {metric: starts_with, equals: The e}...}
        """

        current_line_number = line_obj.get('line_number')
        metric_evaluation_list = []

        line_satisfies_all_metrics = True

        for metric in self.metrics:
            metric_found = False  # Metric does not exist in cache
            for evaluation in stored_evaluations:
                if evaluation.get('metric').to_dict() == metric.to_dict():
                    metric_found = True  # Metric exist in cache
                    if current_line_number not in evaluation.get('passed_lines'):
                        line_satisfies_all_metrics = False

            # Metric does not exist in cache, we cannot
            if not metric_found:
                line_satisfies_all_metrics = False

            metric_evaluation_list.append(line_satisfies_all_metrics)

        if self.condition_type == "AND" and not all(metric_evaluation_list):
            return False

        # Check "OR" conditions: At least one condition in the "OR" list must be true for the line to pass.
        # If the "OR" key is in the identifier and none of the conditions in the "OR" list are true, skip this line.
        if self.condition_type == "OR" and not any(metric_evaluation_list):
            return False

        # Check "NOT" conditions: All conditions in the "NOT" list must be false for the line to pass.
        # If the "NOT" key is in the identifier and any condition in the "NOT" list is true, skip this line.
        if self.condition_type == "NOT" and any(metric_evaluation_list):
            return False

        # Exclusive OR (Only one of the given conditions must be true)
        if self.condition_type == "XOR" and sum(metric_evaluation_list) != 1:
            return False

        # The NAND operation is the negation of the AND operation. It returns True unless both inputs are True.
        if self.condition_type == "NAND" and all(metric_evaluation_list):
            return False

        # The NOR operation is the negation of the OR operation. It returns True only if both inputs are False.
        if self.condition_type == "NOR" and any(metric_evaluation_list):
            return False

        # The XNOR operation is the negation of the XOR (Exclusive OR) operation. It returns True if both inputs are the same (both True or both False).
        if self.condition_type == "XNOR" and sum(metric_evaluation_list) not in [0, len(self.metrics)]:
            return False

        # The next two only excepts two metrics only.

        if self.condition_type == "IMPLICATION":
            # IMPLICATION expects two metrics, metrics[0] implies metrics[1]
            A = metric_evaluation_list[0]
            B = metric_evaluation_list[1]

            if A and not B:  # If first metric is evaluated to True and second is not Implication is failed
                return False

        if self.condition_type == "BICONDITIONAL":
            # BICONDITIONAL expects two metrics, metrics[0] iff metrics[1]
            A = metric_evaluation_list[0]
            B = metric_evaluation_list[1]

            if A != B:
                return False

        return True

    def evaluate(self, line_obj, lines):

        if self.condition_type == "AND" and not all(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Check "OR" conditions: At least one condition in the "OR" list must be true for the line to pass.
        # If the "OR" key is in the identifier and none of the conditions in the "OR" list are true, skip this line.
        if self.condition_type == "OR" and not any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Check "NOT" conditions: All conditions in the "NOT" list must be false for the line to pass.
        # If the "NOT" key is in the identifier and any condition in the "NOT" list is true, skip this line.
        if self.condition_type == "NOT" and any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Exclusive OR (Only one of the given conditions must be true)
        if self.condition_type == "XOR" and sum(metric.evaluate(line_obj, lines) for metric in self.metrics) != 1:
            return False

        # The NAND operation is the negation of the AND operation. It returns True unless both inputs are True.
        if self.condition_type == "NAND" and all(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # The NOR operation is the negation of the OR operation. It returns True only if both inputs are False.
        if self.condition_type == "NOR" and any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # The XNOR operation is the negation of the XOR (Exclusive OR) operation. It returns True if both inputs are the same (both True or both False).
        if self.condition_type == "XNOR" and sum(metric.evaluate(line_obj, lines) for metric in self.metrics) not in [0,
                                                                                                                      len(self.metrics)]:
            return False

        # The next two only excepts two metrics only.

        if self.condition_type == "IMPLICATION":
            # IMPLICATION expects two metrics, metrics[0] implies metrics[1]
            A = self.metrics[0].evaluate(line_obj, lines)
            B = self.metrics[1].evaluate(line_obj, lines)

            if A and not B:  # If first metric is evaluated to True and second is not Implication is failed
                return False

        if self.condition_type == "BICONDITIONAL":
            # BICONDITIONAL expects two metrics, metrics[0] iff metrics[1]
            A = self.metrics[0].evaluate(line_obj, lines)
            B = self.metrics[1].evaluate(line_obj, lines)

            if A != B:
                return False

        return True


