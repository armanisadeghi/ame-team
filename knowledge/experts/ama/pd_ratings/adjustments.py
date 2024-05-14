from .data import age_adjustment_table, occupation_adjustment_table, default_fec


# Adjustments Section ----------------------------------------

# Adjustment for age

def adjust_for_age(occupation_adjusted_rating: int, age: int):
    if age <= 21:
        age_range = "21 and under"
    elif age <= 26:
        age_range = "22 - 26"
    elif age <= 31:
        age_range = "27 - 31"
    elif age <= 36:
        age_range = "32 - 36"
    elif age <= 41:
        age_range = "37 - 41"
    elif age <= 46:
        age_range = "42 - 46"
    elif age <= 51:
        age_range = "47 - 51"
    elif age <= 56:
        age_range = "52 - 56"
    elif age <= 61:
        age_range = "57 - 61"
    else:
        age_range = "62 and over"

    # Find the adjusted rating based on the age range and occupation_adjusted_rating
    adjusted_rating = age_adjustment_table[age_range][
        occupation_adjusted_rating]
    return adjusted_rating


# Adjustment for occupation

def adjust_for_occupation(adjusted_rating: int, occupational_variant: str):
    str_adjusted_rating = str(adjusted_rating)  # Since the keys in occupational variant table are strings
    return occupation_adjustment_table[str_adjusted_rating][occupational_variant]


# Adjustment for FEC rank


def adjust_for_fec(pain_adjusted_wpi: int):
    return round(pain_adjusted_wpi * default_fec)
