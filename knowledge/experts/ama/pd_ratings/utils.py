from datetime import datetime, timedelta, date
from .data import (variant_table,
                  money_chart)


# Combining multiple ratings

def combine_pd_ratings(pd_ratings: list):
    if not pd_ratings:
        return 0
    """
    This function combines multiple PD or other ratings using a specific formula.

    Args:
        pd_ratings: A list of PD ratings (floats between 0 and 100).

    Returns:
        The final combined PD rating (float).
    """

    while len(pd_ratings) > 1:
        # Sort the ratings in descending order
        sorted_ratings = sorted(pd_ratings, reverse=True)

        # Get the largest and second-largest ratings (already percentages)
        largest = sorted_ratings[0] / 100  # Convert to decimal from percentage
        # Convert to decimal from percentage
        second_largest = sorted_ratings[1] / 100

        # Apply the formula a / 100 + b / 100 * (1 - a / 100)
        combined_rating = largest + second_largest * (1 - largest)

        # Round the combined rating to the nearest whole number
        combined_rating = round(combined_rating * 100)

        # Update the list with the combined rating
        pd_ratings = [combined_rating] + sorted_ratings[2:]

    # Return the final combined rating (only element remaining in the list)
    return pd_ratings[0]


# Age handler

def handle_age(date_of_birth=None, date_of_injury=None, age=None):
    if isinstance(date_of_birth, str):
        date_of_birth = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
    if isinstance(date_of_injury, str):
        date_of_injury = datetime.strptime(date_of_injury, "%Y-%m-%d").date()

    if all([date_of_birth, date_of_injury, age]):
        return date_of_birth, date_of_injury, age

    elif not any([date_of_birth, date_of_injury, age]):
        raise ValueError("At least one of date_of_birth, date_of_injury, or age must be provided.")

    elif date_of_birth and date_of_injury:
        age = date_of_injury.year - date_of_birth.year - (
                (date_of_injury.month, date_of_injury.day) < (date_of_birth.month, date_of_birth.day))
        return date_of_birth, date_of_injury, age

    elif date_of_injury and age:
        date_of_birth = date_of_injury - timedelta(days=age * 365.25)
        return date_of_birth, date_of_injury, age

    elif date_of_birth and age:
        date_of_injury = date_of_birth + timedelta(days=(age + 1) * 365.25)
        if date_of_injury > date.today():
            date_of_injury = date.today()
        return date_of_birth, date_of_injury, age


# Find occupational variant

def determine_occupational_variant(group_number: str, impairment_number: str):
    for impairment in variant_table['impairments']:
        if group_number not in impairment['occupationCode']:
            continue

        impairment_category = impairment['category']

        if '--' in impairment_category:  # Range
            start_range, end_range = map(str.strip, impairment_category.split('--'))
            start_major, start_minor = map(int, start_range.split("."))
            end_major, end_minor = map(int, end_range.split("."))
            impairment_items_within_range = [f"{start_major}.{i:02d}" for i in range(start_minor, end_minor + 1)]
            if any(impairment_number.startswith(item) for item in impairment_items_within_range):
                index_no = impairment['occupationCode'].index(group_number)
                return impairment['grade'][index_no]

        elif 'XX' in impairment_category:  # Wildcard
            refined_impairment_category = ".".join(x for x in impairment_category.split('.') if x != "XX")
            if impairment_number.startswith(refined_impairment_category):
                index_no = impairment['occupationCode'].index(group_number)
                return impairment['grade'][index_no]

        elif impairment_category == impairment_number:  # Exact match
            index_no = impairment['occupationCode'].index(group_number)
            return impairment['grade'][index_no]

    return False


def calc_money_chart(pd_percentage: int, year: int, earnings: int):
    weeks = money_chart[pd_percentage]['weeks']

    if year > 2013:
        compensation = weeks * earnings
    else:
        compensation = 1

    return {
        'compensation': compensation,
        "weeks": weeks,
        "days": weeks * 7,
        "print_values": {
            "weeks": f"${weeks:,.2f}",
            "days": f"${weeks * 7:,.2f}",

            "compensation": f"${compensation:,.2f}"
        }
    }
