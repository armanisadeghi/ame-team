from knowledge.experts.ama.pd_ratings.utils import determine_occupational_variant

from .adjustments import (adjust_for_fec,
                         adjust_for_occupation,
                         adjust_for_age,
                         default_fec)

from .data import default_fec_valid_from


def get_single_impairment_rating(
        impairment_number: str,
        wpi_percentage: int,
        occupation_group_code: str,
        age: int,
        injury_year: int,
        industrial_percentage: int,
        pain_percentage: int,
):
    # Initializing Warnings
    warning = []

    wpi_rating_before_pain = wpi_percentage
    wpi_rating_after_pain = wpi_rating_before_pain + pain_percentage

    # Check If the Occupation supplied is Valid:

    if not occupation_group_code:
        raise ValueError('Occupation Code Not Supplied.')

    # Checking if the Occupational variant can be determined
    occupational_variant = determine_occupational_variant(occupation_group_code, impairment_number)

    if not occupational_variant:
        raise ValueError("Cannot Find Occupational Variant")

    # Adjusting for WPI for FEC rank. Using the Modified PDRS 2005,
    # We have to use 1.4 Multiplier for all Injuries on or after 1/1/2013

    if not injury_year > default_fec_valid_from:
        warning.append(f"The injury date is before {default_fec_valid_from}. The accuracy will be affected.")

    adjusted_rating = adjust_for_fec(wpi_rating_after_pain)

    if adjusted_rating > 100:  # As WPI cannot be greater than 100
        adjusted_rating = 100

    # Adjust the rating for occupation
    occupation_adjusted_rating = adjust_for_occupation(adjusted_rating, occupational_variant)

    # Finally adjusting for age
    final_rating = adjust_for_age(occupation_adjusted_rating, age)

    # Industry percentage calculation i.e, apportionment
    final_rating_after_industrial = round(final_rating * industrial_percentage / 100)

    # Display the final permanent disability rating and the rating formula
    formula = f"{impairment_number} - {wpi_percentage} - [{default_fec}]{adjusted_rating} - {occupation_group_code}{occupational_variant} - {occupation_adjusted_rating}% - {final_rating_after_industrial}%"

    return final_rating_after_industrial, formula, warning
