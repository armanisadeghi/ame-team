from knowledge.doc_manipulation.text_classification.text_classifier import (Metric, LineIdentifier)

# Predefined list of metrics
list_metrics = [
    "starts_with",
    "ends_with",
    "contains_regex_pattern",
    "contains",
    "unique_words"
]


def metric_to_db_obj(metric: dict, all_metrics: list):
    metric_type = metric["metric"]
    parameters = {key: metric[key] for key in metric if key != "metric"}
    index = all_metrics.index(metric_type)
    db_metric = {
        "metric_index": index,
        "parameters": parameters
    }

    return db_metric


def db_metric_to_obj(db_metric, all_metrics: list):
    metric_index = db_metric["metric_index"]
    metric = all_metrics[metric_index]
    parameters = db_metric["parameters"]

    metric_dict = {
        "metric": metric,
        **parameters
    }
    return metric_dict


# Example usage
metric_obj = {
    "metric": "starts_with",
    "case_sensitive": True,
    "equals": "Chapter"
}

db_metric = metric_to_db_obj(metric_obj, list_metrics)

# How do we save the results


# let's say we receive this:
# db_metric =  {'metric_index': 0, 'parameters': {'case_sensitive': True, 'equals': 'Chapter'}}

# we will convert something like this to a readable format


chapter_start_metric = db_metric_to_obj(db_metric, list_metrics)

metric_ = Metric(None, None)
metric_.from_dict(chapter_start_metric)

# Now we will automatically evaluate the metric

# metric_.evaluate(None, None)


# We are done with metric stuff we can store the metric now into the db and get the index of the stored metric
# lets say the metric we want to store and evaluate is at 0 index in the metric
# so now:


def evaluate_metric(metric, lines, storage_index):
    passed_lines = []

    for line in lines:
        passes = metric.evaluate(line, lines)
        if passes:
            passed_lines.append(line.get('line_number'))

    return {
        "added_metric": storage_index,
        "lines": passed_lines
    }


# The function above returns a storage object.


# Now , we have to solve how identifiers have to be processed.
# Each condition group has a list of metrics, in the new system those are just indexes of the stored metrics
# So now time to make a function that can interpret an identifier with list of metric indexes

# Incoming A identifier from frontend

stored_metrics = [db_metric]

idf_from_frontend = {
    "0": {
        "type": "AND",
        "metrics": [
            0
        ]
    }
}


# what it means is that we have metrics that are stored at index 0 and 1 inside a array
# interpreting them


def db_identifier_to_obj(db_identifier: dict, stored_metrics: list):
    for k, v in db_identifier.items():
        metric_indices = v.get('metrics')
        v['metrics'] = []

        for m in metric_indices:
            v['metrics'].append(db_metric_to_obj(stored_metrics[m], list_metrics))

    return db_identifier


# Now this converted object can be used as a identifier from dict

idf_dict = db_identifier_to_obj(idf_from_frontend, stored_metrics)

print(idf_dict)
idf_obj = LineIdentifier()
idf_obj.from_dict(idf_dict)


