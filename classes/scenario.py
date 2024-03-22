import numpy as np
import pandas as pd

class Scenario(object):
    """
    Global variables for model.

    class Scenario():

    Attributes
    ----------

    hospitals:
        Info on stroke hospitals. Pandas DataFrame.

    inter_arrival_time:
        Time (minutes) between arrivals. Decimal.

    limit_to_england:
        Limit model to only England admissions. Boolean

    lsoa_names:
        List of LSOA names. List.

    lsoa_relative_frequency:
        Relative frequency of admissions to each LSOA (sums to 1). NumPy array.

    lsoa_ivt_travel_time:
        Travel time (minutes) from LSOA to closest IVT unit. Dictionary.

    lsoa_ivt_unit:
        Closest IVT unit postcode for each LSOA. Dictionary.

    lsoa_mt_travel_time:
        Travel time (minutes) from LSOA to closest MT unit. Dictionary.

    lsoa_mt_unit:
        Closest MT unit postcode for each LSOA. Dictionary.

    mt_transfer_time:
        Time (minutes) for closest IVT to MT transfer. Dictionary.

    mt_transfer_unit:
        Closest MT unit for each IVT unit.  Dictionary.

    process_time_ambulance_response:
        Min/max of time from 999 call to ambulance arrival (tuple of integers)

    run_duration:
        Simulation run time (minutes, including warm-up). Integer.

    total_admissions:
        Total yearly admissions (obtained from LSOA admissions). Float.

    warm_up:
        Simulation run time (minutes) before audit starts.


    Methods
    -------
    load_data:
        Loads data to be used

    _load_hospitals:
        Loads data on the requested stroke teams.

    _load_admissions:
        Loads admissions data for the requested stroke teams and
        the LSOAs in their catchment areas.

    _load_lsoa_travel:
        Loads data on travel times from each LSOA to its nearest
        stroke units offering IVT and MT.

    _load_stroke_unit_travel:
        Loads data on travel times between stroke units.
    """

    def __init__(
            self,
            df_units,
            df_transfer,
            df_lsoa,
            *initial_data,
            **kwargs
            ):
        """
        Constructor method for model parameters

        """
        # ----- Directory setup -----
        # Name that will also be used for output directory:
        self.name = 'scenario'

        # ----- Simpy parameters -----
        # The following batch of parameters are not called anywhere
        # during Scenario() or associated setup classes.
        # They will be called during the Patient, Pathway, and Model
        # classes.
        self.run_duration = 365  # Days
        self.warm_up = 50
        # What are the chances of treatment?
        self.probability_ivt = 1.0
        self.probability_mt = 1.0
        # Set process times.
        # Each tuple contains (minimum time, maximum time).
        # When both values are the same, all generated times
        # are that same value with no variation.
        self.process_time_call_ambulance = (30, 30)
        self.process_time_ambulance_response = (30, 30)
        self.process_ambulance_on_scene_duration = (20, 20)
        self.process_time_arrival_to_needle = (30, 30)
        self.process_time_arrival_to_puncture = (45, 45)
        self.transfer_time_delay = 30
        self.process_time_transfer_arrival_to_puncture = (60, 60)

        # ----- Overwrite default values -----
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Convert run duration to minutes
        self.run_duration *= 1440

        self.keep_only_selected_in_df(df_units, df_transfer, df_lsoa)

        admissions = self.load_admissions()
        self.match_admissions_to_selected_lsoa(admissions)
        self.process_admissions()

    def keep_only_selected_in_df(self, df_units, df_transfer, df_lsoa):
        # Pull out only the "selected" parts of these.
        self.df_selected_units = df_units[
            df_units['selected'] == 1].copy()
        self.df_selected_transfer = df_transfer[
            df_transfer['selected'] == 1].copy()
        self.df_selected_lsoa = df_lsoa[
            df_lsoa['selected'] == 1].copy()

    # ######################
    # ##### ADMISSIONS #####
    # ######################
    def load_admissions(self):
        """
        Load admission data on the selected stroke teams.

        Stores
        ------

        total_admissions:
            float. Total admissions in a year across selected
            stroke units.

        lsoa_relative_frequency:
            np.array. Relative frequency of each considered LSOA
            in the admissions data. Same order as self.lsoa_names.

        lsoa_names:
            np.array. Names of all LSOAs considered.
            Same order as lsoa_relative_frequency.

        inter_arrival_time:
            float. Average time between admissions in the
            considered stroke teams.
        """
        # TO DO - replace with relative import
        # Load and parse admissions data
        # # Relative import from package files:
        # path_to_file = files('catchment.data').joinpath(
        #     'admissions_2017-2019.csv')
        # Load and parse unit data TO DO - change to relative import above
        path_to_file = './data/admissions_2017-2019.csv'
        admissions = pd.read_csv(path_to_file)

        admissions = admissions.rename(columns={'area': 'lsoa'})
        return admissions

    def match_admissions_to_selected_lsoa(self, admissions):
        """
        write me
        """
        # Keep only these LSOAs in the admissions data:
        df_lsoa = self.df_selected_lsoa.copy()
        df_lsoa = df_lsoa.reset_index()
        admissions = pd.merge(left=df_lsoa, right=admissions,
                              on='lsoa', how='left')

        admissions_mask = admissions.loc[admissions['selected'] == 1].copy()

        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        total_admissions = np.round(
            admissions_mask['admissions'].sum(), 0).tolist()

        # Relative frequency of admissions across a year:
        admissions_mask['relative_frequency'] = (
            admissions_mask['admissions'] / total_admissions)

        # Merge this info back into the main DataFrame:
        admissions = pd.merge(
            admissions, admissions_mask[['lsoa', 'relative_frequency']],
            on='lsoa', how='left')

        # Set index to both LSOA name and code so that both follow
        # through to all of the results data.
        admissions = admissions.set_index(['lsoa', 'lsoa_code'])

        self.df_selected_lsoa = admissions

    def process_admissions(self):
        """
        Get some stats from the existing admissions DataFrame.

        DataFrame must have "admissions" column.
        DataFrame must have "relative_frequency" column.
        """
        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        self.total_admissions = np.round(
            self.df_selected_lsoa['admissions'].sum(), 0).tolist()
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions
