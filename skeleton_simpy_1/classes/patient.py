import pandas as pd
import random

class Patient():
    """
    Patient object

    Attributes
    ----------
    closest_ivt_unit:
        Name or postcode of closest IVT unit. Text.

    closest_ivt_time:
        Predicted travel time (minutes) to closest IVT unit. Float.

    closest_mt_unit:
        Name or postcode of closest MT unit. Text.

    closest_mt_time:
        Predicted travel time (minutes) to closest MT unit. Float.

    lsoa:
        Home LSOA. Text.


    mt_transfer_unit:
        Name or postcode of closest MT transfer unit. Text.

    mt_transfer_time:
        Predicted travel time (minutes) to closest MT transfer unit. Float.

    mt_transfer_required:
        Is transfer required for MT? Boolean.

    


    """

    def __init__(self, globvars, id):
        """Constructor class for patient"""

        self.id = id

        # Get LSOA based on admission numbers
        elements = globvars.lsoa_names
        frequencies = globvars.lsoa_relative_frequency
        self.lsoa = random.choices(elements, weights=frequencies)[0]

        # Get unit details
        self.closest_ivt_unit = globvars.lsoa_ivt_unit[self.lsoa]
        self.closest_ivt_time = globvars.lsoa_ivt_travel_time[self.lsoa]
        self.closest_mt_unit = globvars.lsoa_mt_unit[self.lsoa]
        self.closest_mt_time = globvars.lsoa_mt_travel_time[self.lsoa]
        self.mt_transfer_unit = globvars.mt_transfer_unit[self.closest_ivt_unit]
        self.mt_transfer_time = globvars.mt_transfer_time[self.closest_ivt_unit]
        self.mt_transfer_required = self.closest_mt_unit != self.mt_transfer_unit

        # Set up times; these will be relative to onset
        self.time_onset = ''
        self.time_ambulance_called = ''
        self.time_ambulance_arrives = ''

