import csv
import json


def csv_to_fec_json_and_array(csv_path, json_dict_path, json_array_path):
    fec_adjustment_dict = {}
    fec_adjustment_array = []

    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row if present

        for row in csv_reader:
            fec_rank = row[0]  # Assuming the FEC rank is in the first column
            if fec_rank not in fec_adjustment_dict:
                fec_adjustment_dict[fec_rank] = {}

            adjustments = []
            for i, value in enumerate(row[1:], start=1):  # Start from the second column
                fec_adjustment_dict[fec_rank][i] = int(value)
                adjustments.append(int(value))

            fec_adjustment_array.append(adjustments)

    # Saving nested dictionary as JSON
    with open(json_dict_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(fec_adjustment_dict, jsonfile, ensure_ascii=False, indent=4)

    # Saving two-dimensional array as JSON
    with open(json_array_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(fec_adjustment_array, jsonfile, ensure_ascii=False, indent=4)

    return fec_adjustment_dict, fec_adjustment_array


def csv_to_occupation_json(csv_path, json_path):
    occupation_data = []

    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter='\t')  # Assuming the delimiter is a tab
        for row in csv_reader:
            occupation_data.append(row)

    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(occupation_data, jsonfile, ensure_ascii=False, indent=4)

    return occupation_data


if __name__ == '__main__':
    '''
    csv_file_path = 'data/ame_pdr_fec_adjustment_table.csv'
    json_dict_path = 'data/ame_pdr_fec_adjustment_table_dict.json'
    json_array_path = 'data/ame_pdr_fec_adjustment_table_array.json'
    fec_adjustment_dict, fec_adjustment_array = csv_to_fec_json_and_array(csv_file_path, json_dict_path, json_array_path)

    # Printing the two-dimensional array in a more compact format
    print("fec_adjustments_array = [")
    for row in fec_adjustment_array:
        print(f"    {row},")
    print("]")

    print(f"\nJSON dictionary file '{json_dict_path}' and JSON array file '{json_array_path}' have been created.")
    '''
    csv_file_path = 'data/occupation_group_numbers.csv'  # Update this path
    json_file_path = 'data/occupation_group_numbers.json'  # Update this path
    occupation_data = csv_to_occupation_json(csv_file_path, json_file_path)

    # Printing the list of dictionaries
    for occupation in occupation_data:
        print(occupation)

    print(f"\nJSON file '{json_file_path}' has been created.")
