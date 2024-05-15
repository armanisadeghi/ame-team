from .data import (
    thumb_impairment_to_hands_impairment,
    ring_or_little_finger_impairment_to_hand_impairment,
    index_or_middle_finger_impairment_to_hand_impairment,
    hand_impairment_to_upper_extremity)


# Conversion Section ------------------------------------

# Lower Extremity to WPI %


def lower_extremity_to_wpi(le: int):
    if le >= 100:
        return round(100 * 0.4)

    return round(le * 0.4)


# Wpi to lower extremity

def wpi_to_lower_extremity(wpi: int):
    if wpi >= 40:
        return 100

    return round(wpi / 0.4)


# Upper Extremity to WPI %

def upper_extremity_to_wpi(ue: int):
    if ue >= 100:
        return round(100 * 0.6)

    return round(ue * 0.6)


# Wpi to Upper Extremity
def wpi_to_upper_extremity(wpi: int):
    if wpi >= 60:
        return 100

    return round(wpi / 0.6)


# Hand impairment to Upper Extremity

def hand_impairment_to_ue(hand_impairment_percent: int):
    return hand_impairment_to_upper_extremity[hand_impairment_percent]


# Finger to hand impairment calculation

def range_search(digit: int, range_dict: dict):
    for range_str, value in range_dict.items():
        start, end = map(int, range_str.split('-'))
        if start <= digit <= end:
            return value


def finger_impairment_to_hand(finger_type: str, digit_impairment: int):
    """
    finger_type : It should be either of thumb, index, middle, ring or little
    """
    if finger_type not in ['thumb', 'index', 'middle', 'ring', 'little']:
        raise ValueError("Invalid finger type supplied")

    if finger_type == "thumb":
        value = range_search(digit_impairment, thumb_impairment_to_hands_impairment)
    elif finger_type == "index" or finger_type == 'middle':
        value = range_search(digit_impairment, index_or_middle_finger_impairment_to_hand_impairment)
    else:  # Last case if for ring or little finger
        value = range_search(digit_impairment, ring_or_little_finger_impairment_to_hand_impairment)
    return value

# Conversion Section Ends ------------------------------------
