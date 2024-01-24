"""
Scenario class with global parameters for the pathway model.
"""
import numpy as np
import pandas as pd

from classes.units import Units

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

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        # Which LSOAs will we use?
        self.mt_hub_postcodes = []
        self.limit_to_england = True

        self.run_duration = 365  # Days
        self.warm_up = 50

        # Which stroke team choice model will we use?
        self.destination_decision_type = 0
        # 0 is 'drip-and-ship'

        # Are we using any extra units?
        # i.e. not used in the main IVT and MT units list.
        self.custom_units = False

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

        # Stroke unit services updates.
        # Change which units provide IVT, MT, and MSU by changing
        # their 'Use_IVT' flags in the services dataframe.
        # Example:
        # self.services_updates = {
        #     'hospital_name1': {'Use_MT': 0},
        #     'hospital_name2': {'Use_IVT': 0, 'Use_MSU': None},
        #     }
        self.services_updates = {}

        # Set up paths to files.
        # TO DO - check if output folder already exists,
        # make a new output folder for each run.
        self.paths_dict = dict(
            data_read_path='./data/',
            output_folder='./output/',
        )

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Convert run duration to minutes
        self.run_duration *= 1440

        # Load data:
        # (run this after MT hospitals are updated in
        # initial_data or kwargs).
        self.load_data()

    def load_data(self):
        """
        Load required data.

        Stores the following in the Globvars object:
        + hospitals
        + lsoa_names
        + total_admissions
        + lsoa_relative_frequency
        + inter_arrival_time
        + lsoa_ivt_travel_time
        + lsoa_ivt_unit
        + lsoa_mt_travel_time
        + lsoa_mt_unit
        + lsoa_msu_travel_time
        + lsoa_msu_unit
        + mt_transfer_time
        + mt_transfer_unit

        More details on each attribute are given in the docstrings
        of the methods that create them.
        """
        # ##### NATIONAL UNITS #####
        units = Units({'services_updates': self.services_updates})
        self.national_dict = units.load_data()

        # ##### SELECTED UNITS #####
        # Import hospital names:
        self._load_hospitals()
        # Stores:
        # + self.hospitals

        # Find which LSOAs are in these stroke teams' catchment areas:
        self._load_lsoa_names()
        # Stores:
        # + self.lsoa_names

        # Import admissions statistics for those hospitals:
        self._load_admissions()
        # Stores:
        # + self.total_admissions
        # + self.lsoa_relative_frequency
        # + self.inter_arrival_time

        # Stroke unit data for each LSOA.
        self._load_lsoa_travel()
        # Stores:
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit
        # + self.lsoa_mt_travel_time
        # + self.lsoa_mt_unit
        # + self.lsoa_msu_travel_time
        # + self.lsoa_msu_unit


    # ##########################
    # ##### SELECTED UNITS #####
    # ##########################
    def _load_hospitals(self):
        """
        Load data on the selected stroke units.

        If no units are specified but "limit_to_england" is True,
        then only English stroke units are kept. If no units are
        specified and "limit_to_england" is False, then all stroke
        units are kept.

        Stores
        ------

        hospitals:
            pd.DataFrame. Each stroke team's data including name,
            postcode, region, lat/long, provide IVT or MT...
        """
        # Load and parse hospital data
        hospitals = pd.read_csv("./data/stroke_hospitals_2022.csv")
        # Only keep stroke units that offer IVT and/or MT:
        hospitals["Use"] = hospitals[["Use_IVT", "Use_MT"]].max(axis=1)
        mask = hospitals["Use"] == 1
        hospitals = hospitals[mask]

        # Limit the available hospitals if required.
        if len(self.mt_hub_postcodes) > 0:
            # If a list of MT units was given, use only those units.

            # Which IVT units are feeder units for these MT units?
            # These lines also pick out the MT units themselves.
            df_feeders = pd.read_csv('./data/nearest_mt_each_hospital.csv')
            # For each MT unit, find whether each stroke unit has this
            # as its nearest MT unit. Get a long list of True/False
            # values for each MT unit.
            feeders_bool = [
                df_feeders['name_nearest_MT'].str.contains(s)
                for s in self.mt_hub_postcodes
                ]
            # Make a mask that is True for any stroke unit that
            # answered True to any of the feeders_bool lists,
            # i.e. its nearest MT unit is in the requested list.
            mask = np.any(feeders_bool, axis=0)
            # Limit the feeders dataframe to just those stroke teams
            # to get a list of postcodes of feeder units.
            df_feeders = df_feeders[mask]
            # Limit the "hospitals" dataframe to these feeder postcodes:
            hospitals = pd.merge(
                left=hospitals,
                right=df_feeders,
                left_on='Postcode',
                right_on='from_postcode',
                how='inner'
            )[hospitals.columns]
        elif self.limit_to_england:
            # Limit the data to English stroke units only.
            mask = hospitals["Country"] == "England"
            hospitals = hospitals[mask]
        else:
            # Use the full "hospitals" data.
            pass
        self.hospitals = hospitals

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'selected_stroke_units.csv'
        hospitals.to_csv(f'{dir_output}{file_name}')

    def _load_lsoa_names(self):
        """
        WIP
        Load names of LSOAs in catchment areas of chosen stroke teams.

        Stores
        ------
        lsoa_names:
            np.array. Names of all LSOAs considered.
        """
        # Take list of all LSOA names and travel times:
        df_travel = self.national_dict['lsoa_nearest_units']
        # This has one row for each LSOA nationally and columns
        # for LSOA name and ONS code (LSOA11NM and LSOA11CD),
        # and time, postcode, and SSNAP name of the nearest unit
        # for each unit type (IVT, MT, MSU).
        # Columns:
        # + LSOA11NM
        # + LSOA11CD
        # + time_nearest_IVT
        # + postcode_nearest_IVT
        # + ssnap_name_nearest_IVT
        # + time_nearest_MT
        # + postcode_nearest_MT
        # + ssnap_name_nearest_MT
        # + time_nearest_MSU
        # + postcode_nearest_MSU
        # + ssnap_name_nearest_MSU

        # Limit the available hospitals if required.
        if len(self.mt_hub_postcodes) > 0:
            # If a list of MT units was given, limit the admissions
            # data to just those MT units and feeder IVT units
            # that we saved earlier as self.hospitals.

            # Which LSOAs are in the catchment areas for these IVT units?
            # For each stroke team, make a long list of True/False for
            # whether each LSOA has this as its nearest unit.
            postcode_cols = [
                'postcode_nearest_IVT',
                'postcode_nearest_MT',
                'postcode_nearest_MSU',
            ]
            lsoa_bool = [
                df_travel[col].str.contains(s)
                for col in postcode_cols
                for s in self.hospitals['Postcode'].values
                ]
            # Mask is True for any LSOA that is True in any of the
            # lists in lsoa_bool.
            mask = np.any(lsoa_bool, axis=0)
            print(mask)
            # Limit the data to just these LSOAs:
            lsoas_to_include = df_travel['LSOA11NM'][mask]
        elif self.limit_to_england:
            # Limit the data to English LSOAs only.
            # The LSOA11CD (ONS code for each LSOA) begins with
            # an "E" for English and "W" for Welsh LSOAs.
            # All other characters are numbers.
            mask_england = df_travel['LSOA11CD'].str.contains('E')
            lsoas_to_include = df_travel['LSOA11NM'][mask_england]
        else:
            # Just use all LSOAs in the file.
            lsoas_to_include = df_travel['LSOA11NM']

        # Store in self:
        self.lsoa_names = lsoas_to_include

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'selected_lsoas.csv'
        lsoas_to_include.to_csv(f'{dir_output}{file_name}')

    def _load_admissions(self):
        """
        Load admission data on the selected stroke teams.

        If no units are specified but "limit_to_england" is True,
        then only English stroke units are kept. If no units are
        specified and "limit_to_england" is False, then all stroke
        units are kept.

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
        # Load and parse admissions data
        admissions = pd.read_csv("./data/admissions_2017-2019.csv")

        # Keep only these LSOAs in the admissions data:
        admissions = pd.merge(
            left=admissions,
            right=self.lsoa_names,
            left_on='area',
            right_on='LSOA11NM',
            how='inner'
        )

        # Process admissions.
        # Total admissions across these hospitals in a year:
        self.total_admissions = np.round(admissions["Admissions"].sum(), 0)
        # Relative frequency of admissions across a year:
        self.lsoa_relative_frequency = np.array(
            admissions["Admissions"] / self.total_admissions
        )
        # Overwrite this to make sure the LSOA names are in the
        # same order as the LSOA relative frequency array.
        self.lsoa_names = list(admissions["area"])
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions

    def _load_lsoa_travel(self):
        """
        WIP
        Stroke unit data for each LSOA.

        Stores
        ------

        lsoa_ivt_travel_time:
            dict. Each LSOA's nearest IVT unit travel time.

        lsoa_ivt_unit:
            dict. Each LSOA's nearest IVT unit name.

        lsoa_mt_travel_time:
            dict. Each LSOA's nearest MT unit travel time.

        lsoa_mt_unit:
            dict. Each LSOA's nearest MT unit name.
        """
        # Use the list of LSOA names to include:
        lsoa_names = self.lsoa_names

        # Take list of all LSOA names and travel times:
        df_travel = self.national_dict['lsoa_nearest_units']
        # This has one row for each LSOA nationally and columns
        # for LSOA name and ONS code (LSOA11NM and LSOA11CD),
        # and time, postcode, and SSNAP name of the nearest unit
        # for each unit type (IVT, MT, MSU).
        # Columns:
        # + LSOA11NM
        # + LSOA11CD
        # + time_nearest_IVT
        # + postcode_nearest_IVT
        # + ssnap_name_nearest_IVT
        # + time_nearest_MT
        # + postcode_nearest_MT
        # + ssnap_name_nearest_MT
        # + time_nearest_MSU
        # + postcode_nearest_MSU
        # + ssnap_name_nearest_MSU

        # Limit the big DataFrame to just the LSOAs wanted:
        df_travel = pd.merge(
            df_travel,
            pd.DataFrame(lsoa_names, columns=['LSOA11NM']),
            left_on='LSOA11NM',
            right_on='LSOA11NM'
            )
        df_travel = df_travel.set_index('LSOA11NM')

        # Separate out the columns and store in self:
        self.lsoa_ivt_travel_time = dict(df_travel['time_nearest_IVT'])
        self.lsoa_ivt_unit = dict(df_travel['postcode_nearest_IVT'])
        self.lsoa_mt_travel_time = dict(df_travel['time_nearest_MT'])
        self.lsoa_mt_unit = dict(df_travel['postcode_nearest_MT'])
        self.lsoa_msu_travel_time = dict(df_travel['time_nearest_MSU'])
        self.lsoa_msu_unit = dict(df_travel['postcode_nearest_MSU'])
