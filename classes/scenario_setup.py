import numpy as np

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

        # Does this need a Setup() object? Can we get away without one?

        # Basic idea here -
        # import existing dataframes for units, regions(?), transfer units, admissions
        # and then turn them into the dictionaries etc that the pathway model needs.
        # Don't actually need regions.
        # Unit info quietly in LSOA info? + transfre units?

    # #############################
    # ##### SETUP FOR PATHWAY #####
    # #############################
    def create_lsoa_travel_dicts(self, df_lsoa):
        """
        Convert LSOA travel time dataframe into separate dicts.

        DataFrame requires columns:
        """
        # Now create dictionaries of the LSOA travel times.
        # TO DO - make this updateable for MT, MSU -------------------------------------
        # ..?

        # So don't bother with dicts here either?


        # Separate out the columns and store in self:
        travel_key = f'lsoa_unit_travel_time'
        travel_val = df_lsoa['unit_travel_time']
        setattr(self, travel_key, travel_val)

        unit_key = f'lsoa_unit_postcode'
        unit_val = df_lsoa['unit_postcode']
        setattr(self, unit_key, unit_val)

    def make_transfer_dicts(self, df_selected_transfer_units):
        # TO DO
        # # Patient class expects:
        df_selected_transfer_units
        # Index: postcode
        # Columns: name_nearest_mt

        # BIG TO DO - what to do about separate IVT, MT, MSU calculations? ----------
        # Nearest MT unit vs transfer MT unit?
        # Then selecting whether to go to nearest MT or nearest IVT...
        # worth storing the info in the dataframes at least.
        # Dataframe and at()?
        # docs claim that df1.loc['a', 'A'] is equivalent to df1.at['a','A'].
        # Both seem to return scalars. When only one column and one row.

        # Switch to DataFrames?
        # Then don't need this Scenario to do much setup...

        # # TO DO - currently piggybacking off transfer units.
        # # Change to separate MT unit?
        # self.closest_mt_unit = scenario.national_dict['mt_transfer_unit'][self.closest_ivt_unit]
        # self.closest_mt_travel_duration = (
        #     scenario.national_dict['mt_transfer_time'][self.closest_ivt_unit])

        # Should the Patient class accept DataFrames or only dicts? Why would it matter?
        # Dicts means repeating the index column over and over.
        # Dict also means less chance of error, accidentally setting a value as a Series instead of a float.

        # self.mt_transfer_unit = df_selected_transfer_units.loc[
        #     self.closest_ivt_unit, 'name_nearest_mt']
        # self.mt_transfer_travel_duration = df_selected_transfer_units.loc[
        #     self.closest_ivt_unit, 'time_nearest_mt']
        # self.mt_transfer_required = (
        #     self.closest_ivt_unit != self.closest_mt_unit)
        pass

    def process_admissions(self, df_lsoa):
        """
        Get some stats from the existing admissions DataFrame.

        DataFrame must have "admissions" column.
        DataFrame must have "relative_frequency" column.
        """
        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        self.total_admissions = np.round(
            df_lsoa["admissions"].sum(), 0).tolist()
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions
