"""
Patient class for storing stroke pathway details of a single patient.
"""
import random
import numpy as np

# For type hinting:
from classes.scenario import Scenario


class Patient():
    """
    Patient object

    class Patient():

    Attributes
    ----------

    admitting_unit:
        str. Name of the first unit that the patient arrives at.

    admitting_unit_travel_duration:
        float. Travel time to admitting unit.

    closest_ivt_unit:
        str. Name or postcode of closest IVT unit.

    closest_ivt_travel_duration:
        float. Predicted travel time (minutes) to closest IVT unit.

    closest_mt_unit:
        str. Name or postcode of closest MT unit.

    closest_mt_travel_duration:
        float. Predicted travel time (minutes) to closest MT unit.

    id:
        str or int. Patient identity.

    lsoa:
        str. Home LSOA.

    mt_transfer_unit:
        str. Name or postcode of closest MT transfer unit.

    mt_transfer_travel_duration:
        float. Predicted travel time (minutes) to closest MT transfer unit.

    mt_transfer_required:
        bool. Is transfer required for MT?

    stroke_type:
        int. Code number for which stroke type this is. Options are:
        + 0 - haemorrhage
        + 1 - nLVO
        + 2 - LVO
        + 3 - mimic

    thrombectomy:
        bool. Whether this patient receives thrombectomy.

    thrombolysis:
        bool. Whether this patient receives thrombolysis.

    time_ambulance_arrival:
        float. Time from this patient's onset to ambulance arrival.

    time_ambulance_called:
        float. Time from this patient's onset to ambulance being called.

    time_ambulance_leaves_scene:
        float. Time the ambulance leaves the scene of onset.

    time_admitting_unit_arrival:
        float. Time from this patient's onset to arrival at admitting unit.

    time_needle:
        float. Time from this patient's onset to start of thrombolysis.

    time_onset:
        float. Time from simulation start to this patient's onset.

    time_puncture:
        float. Time from this patient's onset to start of thrombectomy.

    time_transfer_unit_arrival:
        float. Time from this patient's onset to arrival at transfer unit.

    Methods
    -------

    __init__:
        Constructor class for patient.

    choose_lsoa:
        Randomly assign an LSOA to this patient.

    generate_stroke_type:
        Randomly assign a stroke type to this patient.
    """

    def __init__(
            self,
            scenario: type[Scenario],
            id: type[str]
            ):
        """
        Constructor class for patient
        """

        p = np.NaN  # Placeholder value for attributes.

        self.id = id

        # Choose LSOA based on admission numbers
        self.lsoa = self.choose_lsoa(scenario)

        # Select stroke type:
        self.stroke_type = self.generate_stroke_type()

        # Find unit details for this LSOA.
        self.unit = scenario.lsoa_unit[self.lsoa]
        self.travel_duration = (
            scenario.lsoa_travel_time[self.lsoa])

        self.mt_transfer_unit = (
            scenario.national_dict['mt_transfer_unit'][self.unit])
        self.mt_transfer_travel_duration = (
            scenario.national_dict['mt_transfer_time'][self.unit])
        self.mt_transfer_required = (
            self.unit != self.mt_transfer_unit)

        # These will be selected later:
        self.admitting_unit = ''
        self.admitting_unit_travel_duration = p
        self.thrombolysis = False
        self.thrombectomy = False

        # Onset time relative to start of simulation:
        self.time_onset = p
        # Set up times; these will be relative to onset
        self.time_ambulance_called = p
        self.time_ambulance_arrival = p
        self.time_ambulance_leaves_scene = p
        self.time_admitting_unit_arrival = p
        self.time_needle = p
        self.time_transfer_unit_arrival = p
        self.time_puncture = p

    def choose_lsoa(self, scenario: type[Scenario]):
        """
        Assign an LSOA to this patient based on admission frequency.

        Returns
        -------

        lsoa:
            str. The chosen LSOA.
        """
        # From the scenario object, take the LSOA names...
        elements = scenario.lsoa_names
        # ... and the frequency of each LSOA in the admission numbers...
        frequencies = scenario.lsoa_relative_frequency
        # ... and make a weighted selection of LSOA.
        lsoa = random.choices(elements, weights=frequencies)[0]
        return lsoa

    def generate_stroke_type(self):
        """
        Randomly assign a stroke type to this patient.

        First decide whether this patient is a mimic.
        If not, then decide whether they have a haemorrhage.
        If not, then decide whether they have a large or non-large
        vessel occlusion (LVO or nLVO).

        Returns
        -------

        stroke_type_code:
            int. Code number for which stroke type this is. Options are:
            + 0 - haemorrhage
            + 1 - nLVO
            + 2 - LVO
            + 3 - mimic
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
