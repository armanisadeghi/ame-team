from .data import (
    impairments_data,
    max_weekly_earnings,
    default_occupation_code_if_not_supplied, default_digit_impairment_if_not_supplied, default_wpi_if_not_supplied
)

from .conversions import (hand_impairment_to_ue,
                          finger_impairment_to_hand,
                          upper_extremity_to_wpi, wpi_to_upper_extremity,
                          lower_extremity_to_wpi, wpi_to_lower_extremity)

from knowledge.experts.ama.pd_ratings.utils import (combine_pd_ratings,
                                                    handle_age, determine_occupational_variant,
                                                    calc_money_chart, get_impairment_config)

from .calculate_single_injury import get_single_impairment_rating
from pprint import pprint
from common import vcprint, pretty_print, cool_print


def sort_impairments(factors, config, categories, index):
    """Sorts impairment data into appropriate categories."""
    side = config.get('side')
    if side not in ["Right", "Left"]:
        side = "Default"

    if factors['digit'] and side in ["Right", "Left"]:  # Now ensuring side is either Right or Left for digits
        categories['digit'][side].append(
            (config['impairment_number'], config.get('digit'), factors.get('digit')[1], index))

    elif factors['ue']:
        categories['ue'][side].append((config['impairment_number'], config.get('ue'), index))

    elif factors['le']:
        categories['le'][side].append((config['impairment_number'], config.get('le'), index))

    else:
        categories['no_extremity'][side].append((config['impairment_number'], config.get('wpi'), index))


def categorize_impairments(impairment_numbers):
    """Categorizes impairments into digits, upper extremities (UE), lower extremities (LE), and no extremity."""
    categories = {
        "digit": {
            "Right": [],
            "Left": []
        },
        "ue": {
            "Right": [],
            "Left": [],
            "Default": []
        },
        "le": {
            "Right": [],
            "Left": [],
            "Default": []
        },
        "no_extremity": {
            "Right": [],
            "Left": [],
            "Default": []
        }
    }

    for index, config in impairment_numbers.items():
        impairment_digits = config['impairment_number']
        default_config = impairments_data[impairment_digits]
        factors = default_config[2]

        sort_impairments(factors, config, categories, index)
    return categories


def process_digit_impairments(digit_impairments, ue_impairments):
    """Converts finger impairments to hand impairments and aggregates them."""

    if digit_impairments:
        hand_impairments = [finger_impairment_to_hand(imp[2], imp[1]) for imp in digit_impairments]
        total_hand_impairment = combine_pd_ratings(hand_impairments)
        hand_ue = hand_impairment_to_ue(total_hand_impairment)
        ue_impairments.append((digit_impairments[0][0], hand_ue, digit_impairments[0][-1]))


def process_side_impairments(impairment_types, side, results):
    """Processes and converts impairments by side."""
    for impairment_type in ["ue", "le", "no_extremity"]:
        for imp in impairment_types[impairment_type][side]:
            wpi = upper_extremity_to_wpi(imp[1]) if impairment_type == "ue" else lower_extremity_to_wpi(
                imp[1]) if impairment_type == "le" else imp[1]
            results[side].append((imp[0], wpi, imp[2]))


def get_combined_ratings(results, age, occupation_code, injury_year, impairment_numbers: dict):
    """Display results and calculates total compensation."""

    combined_data = {
        "Right": [],
        "Left": [],
        "Default": []
    }
    for side, impairments in results.items():

        if impairments:

            for i in impairments:
                impairment_number = i[0]
                wpi = i[1]
                index = i[-1]

                rating, formula, warning = get_single_impairment_rating(impairment_number, wpi, occupation_code, age,
                                                                        injury_year,
                                                                        impairment_numbers[index]['industrial'],
                                                                        impairment_numbers[index]['pain'])
                combined_data[side].append({
                    "rating": rating,
                    "formula": formula,
                    "warning": warning
                })

    return combined_data


def process_combined_ratings_to_single_rating(combined_ratings: dict):
    output = {
        "Right": 0,
        "Left": 0,
        "Default": 0,
        "All": 0
    }
    for side, impairments in combined_ratings.items():
        output[side] = combine_pd_ratings([x['rating'] for x in impairments])

    output['All'] = combine_pd_ratings([output['Right'], output['Left'], output['Default']])

    return output


def find_errors_for_occupation_code(impairment_numbers: dict, occupation_code: str, errors: list):
    """
    Tries to find occupation variant for all the injuries supplied and add a key , i.e. occupation_variant
    """
    # Checking if occupation code can be determined for all injuries

    for index, impairment_config in impairment_numbers.items():
        variant = determine_occupational_variant(occupation_code, impairment_config['impairment_number'])

        if not variant:
            errors.append(
                f"Cannot determine occupation variant {impairment_config['impairment_number']}, Group : {occupation_code}")
        else:
            impairment_config['occupation_variant'] = variant


def adjust_incorrect_impairment_config(impairment_numbers: dict, warnings: list):
    for index, impairment_config in impairment_numbers.items():
        impairment_number = impairment_config.get('impairment_number')
        impairment_requirements = get_impairment_config(impairment_number)
        if impairment_requirements is not None:
            impairment_inputs = impairment_requirements[2]  # Since 3rd item is the inputs config

            # Adjusting if WPI is not present when required
            if impairment_inputs.get('wpi'):
                if impairment_config.get(
                        'wpi') is None:  # Say for any case wpi is None, we will set a default value here
                    # not a very ideal case but still to avoid errors
                    impairment_numbers[index]['wpi'] = default_wpi_if_not_supplied

                    # If UE or LE is present in this case , and wpi is not supplied for that reason the default wpi value won't affect further calculation
                    # this only affects the calculation where LE and UE are required

            # Adjusting if UE is not present when required
            if impairment_inputs.get('ue'):
                if impairment_config.get('ue') is None and impairment_config.get('wpi'):
                    impairment_numbers[index]['ue'] = wpi_to_upper_extremity(impairment_config.get('wpi'))

            # Adjusting if LE is not present when required
            if impairment_inputs.get('le'):
                if impairment_config.get('le') is None and impairment_config.get('wpi'):
                    impairment_numbers[index]['le'] = wpi_to_lower_extremity(impairment_config.get('wpi'))

            # Adjusting if Side is not present when required
            if impairment_inputs.get('side'):
                if impairment_config.get('side') is None or impairment_config.get('side') not in ['Right', 'Left']:
                    impairment_numbers[index][
                        'side'] = 'Right'  # Defaulting the side to right since there are only sides
                    warnings.append(f'Side not supplied for {impairment_number} , taking default value: Right')

            # Adjusting digit impairment if not present when required
            if impairment_inputs.get('digit'):
                if impairment_config.get('digit') is None:
                    impairment_numbers[index]['digit'] = default_digit_impairment_if_not_supplied
                    warnings.append(
                        f'Digit impairment not supplied for {impairment_number} ,taking default value :{default_digit_impairment_if_not_supplied}')


def pd_rating_orchestrator(impairment_numbers: dict,
                           age,
                           date_of_birth,
                           date_of_injury,
                           occupation_code,
                           weekly_earnings=max_weekly_earnings):
    errors = []
    general_warnings = []

    # adjusting impairment input factors if values are missing
    adjust_incorrect_impairment_config(impairment_numbers, general_warnings)

    if occupation_code is None:
        occupation_code = default_occupation_code_if_not_supplied
        general_warnings.append(f"Occupation code was not supplied, taking default code {occupation_code}")

    # Validating age related things

    try:
        dob, doi, age = handle_age(date_of_birth, date_of_injury, age)
    except Exception as e:
        errors.append(e)

    # Checking if occupation code can be determined for all injuries

    find_errors_for_occupation_code(impairment_numbers=impairment_numbers, occupation_code=occupation_code,
                                    errors=errors)

    # If there are errors stop execution
    if errors:
        return {
            "status": False,
            "errors": errors
        }

    injury_year = doi.year

    impairment_types = categorize_impairments(impairment_numbers)

    # Process digit impairments
    for side in ["Right", "Left"]:
        process_digit_impairments(impairment_types['digit'][side], impairment_types['ue'][side])

    # Convert UE and LE to WPI and accumulate results by side
    results = {
        "Right": [],
        "Left": [],
        "Default": []
    }

    for side in results.keys():
        process_side_impairments(impairment_types, side, results)

    combined_results = get_combined_ratings(results, age, occupation_code, injury_year, impairment_numbers)

    final_output = process_combined_ratings_to_single_rating(combined_results)

    compensation = calc_money_chart(final_output['All'], injury_year, weekly_earnings)

    return {
        "status": True,
        "warnings": general_warnings,
        "detailed_view": combined_results,
        "combined_view": final_output,
        "compensation": compensation
    }


if __name__ == '__main__':
    # Sample input values
    impairments_numbers = {0: {
        'impairment_number': '16.01.02.04',
        'side': 'Right',
        'ue': None,
        'wpi': 9,
        'digit': None,
        'le': None,
        'industrial': 89,
        'pain': 50}, }

    impairments_numbers_2 = {
        0: {'impairment_number': '15.01.01.00', 'side': False, 'ue': False, 'wpi': 16, 'digit': False, 'le': False,
            'industrial': 69, 'pain': 0},
        1: {'impairment_number': '15.02.01.00', 'side': False, 'ue': False, 'wpi': 8, 'digit': False, 'le': False,
            'industrial': 100, 'pain': 3},
        2: {'impairment_number': '15.03.01.00', 'side': False, 'ue': False, 'wpi': 13, 'digit': False, 'le': False,
            'industrial': 62, 'pain': 0},
        3: {'impairment_number': '17.05.03.00', 'side': 'Left', 'ue': False, 'wpi': 3, 'digit': False, 'le': 8,
            'industrial': 70, 'pain': 0},
        4: {'impairment_number': '17.05.03.00', 'side': 'Right', 'ue': False, 'wpi': 3, 'digit': False, 'le': 8,
            'industrial': 70, 'pain': 0}}

    results = pd_rating_orchestrator(
        impairment_numbers=impairments_numbers_2,
        age=None,
        date_of_birth='1972-01-01',
        date_of_injury='2023-01-01',
        occupation_code='310',
        weekly_earnings=500
    )

    pretty_print(data=results, title="PD Rating Calculation Results", color="blue", style="bold")
