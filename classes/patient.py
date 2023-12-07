import pandas as pd
import random
import numpy as np

class Patient():
    """
    Patient object

    Attributes
    ----------
    closest_ivt_unit:
        Name or postcode of closest IVT unit. Text.

    closest_ivt_duration:
        Predicted travel time (minutes) to closest IVT unit. Float.

    closest_mt_unit:
        Name or postcode of closest MT unit. Text.

    closest_mt_time:
        Predicted travel time (minutes) to closest MT unit. Float.

    lsoa:
        Home LSOA. Text.

    mt_transfer_unit:
        Name or postcode of closest MT transfer unit. Text.

    mt_transfer_duration:
        Predicted travel time (minutes) to closest MT transfer unit. Float.

    mt_transfer_required:
        Is transfer required for MT? Boolean.
    """

    def __init__(self, scenario, id):
        """Constructor class for patient"""

        p = np.NaN  # Placeholder value for attributes.

        self.id = id

        # Get LSOA based on admission numbers
        elements = scenario.lsoa_names
        frequencies = scenario.lsoa_relative_frequency
        self.lsoa = random.choices(elements, weights=frequencies)[0]

        # Select stroke type:
        self.stroke_type = self.generate_stroke_type()

        # Get unit details
        self.closest_ivt_unit = scenario.lsoa_ivt_unit[self.lsoa]
        self.closest_ivt_duration = scenario.lsoa_ivt_travel_time[self.lsoa]
        self.closest_mt_unit = scenario.lsoa_mt_unit[self.lsoa]
        self.closest_mt_duration = scenario.lsoa_mt_travel_time[self.lsoa]
        self.mt_transfer_unit = scenario.mt_transfer_unit[
            self.closest_ivt_unit]
        self.mt_transfer_duration = scenario.mt_transfer_time[
            self.closest_ivt_unit]
        self.mt_transfer_required = (
            self.closest_mt_unit != self.mt_transfer_unit)
        # This will be selected later:
        self.first_unit = ''
        self.duration_ambulance_first_unit = p
        self.thrombolysis = False

        # Set up times; these will be relative to onset
        self.time_onset = p
        self.time_ambulance_called = p
        self.time_ambulance_arrives = p
        self.time_ambulance_on_scene = p
        self.time_unit_arrival = p
        # self.time_scan = p
        self.time_needle = p
        self.time_transfer_unit_arrival = p
        self.time_puncture = p

    def generate_stroke_type(self):
        """
        Randomly choose the stroke type.
        """
        # Decide whether the patient is a mimic.
        prob_mimic = 0.33
        mimic = np.random.binomial(1, prob_mimic)
        if mimic == 1:
            # This patient is a mimic.
            stroke_type_code = 3
            return stroke_type_code
        else:
            # This patient is not a mimic.
            pass

        # If the patient is not a mimic, now do an independent
        # check on whether the patient is haemorrhagic or ischaemic.
        prob_haemo = 0.136
        haemo = np.random.binomial(1, prob_haemo)
        if haemo == 1:
            # This patient has a haemorrhage.
            stroke_type_code = 0
            return stroke_type_code
        else:
            # This patient has an ischaemic stroke.
            pass

        # If the patient has an ischaemic stroke, do another
        # independent check of whether this is a large or non-large
        # vessel occlusion.
        prob_lvo = 0.326
        lvo = np.random.binomial(1, prob_lvo)
        if lvo == 1:
            # This patient has a large vessel occlusion.
            stroke_type_code = 2
            return stroke_type_code
        else:
            # This patient has a non-large vessel occlusion.
            stroke_type_code = 1
            return stroke_type_code
