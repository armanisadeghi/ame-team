import json
import os
import random
import knowledge.doc_manipulation.text_classification.text_classifier

# Simulation process

# Step 1 : As soon a document is uploaded we create some default json files for it
#

with open('ama_raw_text.txt', encoding='utf-8') as f:
    document = f.read()

lines = document.split('\n')

# Creating Document folders for storing data for it

folder_name = f"Document_{random.randint(1000, 10000)}"

os.mkdir(folder_name)

# Storing first 100 lines
with open(f'{folder_name}/lines.json', 'w') as f:
    f.write(json.dumps((lines[:100])))

with open(f'{folder_name}/added_metrics.json', 'w') as f:
    f.write(json.dumps(([])))

with open(f'{folder_name}/edits.json', 'w') as f:
    f.write(json.dumps(([])))

with open(f'{folder_name}/identifiers.json', 'w') as f:
    f.write(json.dumps(([])))

with open(f'{folder_name}/metric_results.json', 'w') as f:
    f.write(json.dumps(([])))

with open(f'{folder_name}/searches.json', 'w') as f:
    f.write(json.dumps(([])))

# Step 2:  lets say user adds metrics by looking for things
with open(f'{folder_name}/added_metrics.json', 'r') as f:
    added = json.loads(f.read())

with open(f'{folder_name}/added_metrics.json', 'w') as f:
    # Let's say user choose index 1 from all the metrics.
    # Now index  1 is  endswith if you look into the file all_metrics.json carefully
    # Time to add it to our records

    added.append({
        "metric_index": 1,
        "name": "Check lines that end with 'Impairment'",
        "parameters": {
            "equals": "Impairment",
            "case_sensitive": False
        }
    })
    current_idx = len(added) - 1
    f.write(json.dumps(added))

    # We can keep adding more metrics for this document, The evaluations will automatically be handled. in this case im going to manually add it

    with open(f'{folder_name}/metric_results.json', 'r') as f:
        results = json.loads(f.read())

    with open(f'{folder_name}/metric_results.json', 'w') as f:
        results.append(
            {
                "added_metric": current_idx,
                "line_indexes": [10, 20, 30, 40]
            }
        )
        results = f.write(json.dumps(results))

# Step 3: We can store the identifiers. in this way

with open(f'{folder_name}/identifiers.json', 'r') as f:
    idfs = json.loads(f.read())

with open(f'{folder_name}/identifiers.json', 'w') as f:
    idfs.append({
        "name": "My idf1",
        "comments": "what it is for",
        "identifier": {
            "0": {
                "type": "AND",
                "metrics": [0]  # This line is a list of indexes in added_metrics
            }
        }
    })
    f.write(json.dumps(idfs))


# Step 4: search for things. How this fill work is getting line ranges or lines
# The result of the class FindOperations will look something like this list[int] whereas line ranges will look something like this list[list[int], list[int], list[int] , ....]

search = {
            'name': 'name_this_search',
            "action": 'find_lines_by_identifier',
            'kwargs': {'identifier': 0, 'start_marker': '--somethings--'}  # The params that require LineIdentifier, we can pass the index of it directly while talking to backend
        }

# So the searches will be saved in the above-mentioned format , inside  array in searches.json


# Step 5: User can make edits. We cant really have the user add any text anywhere in the rich text editor. Incase we want to add line at A index ,we will use functions
# from LineOperations class in text_classifier class

edit1 = {
    "action" : 'add_lines_after',
    "kwargs" : {"line_numbers": [1,3,4,5,5], 'text': 'Something'}
}
edit2 = {
    "action" : 'add_lines_before',
    "kwargs" : {"line_numbers": [1,3,4,5,5], 'text': 'Something'}
}

# So in any case the user wants to add a custom line , we have to add an edit object. Typically, an add_line_after edit.
# Please note the edits are applied in order,
# This is important ,because we are not actually editing things, or applying things. We are only "REMEMBERING" to apply these things.
# The way it should be handled on frontend is like this:
# Lets say the user added text before 100 lines. , now on frontend , these lines will be show as green highlighted.
# like in between some text we have a green line that "DOESN'T HAVE ANY INDEX". We have to show the user that you are making some addition
# In the same way when we replace and move stuff or delete stuff we have to show it . Just dont add index to it
# This also means that metric cannot identify these lines as they are not originally in the original document

# Now when the user clicks on make changes, if you can add the javascript to copy the new text from the Editor, i assume it will be not be easy,
# so i can handle it on backend, making edits.

# All this journey was for Round 1 , Now I took a small examples . But in real cases it can be more that 100 edits each time.
# Now we make repeat the above steps if the user want to start fresh from round 1

