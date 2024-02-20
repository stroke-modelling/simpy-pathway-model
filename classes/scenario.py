"""
Scenario class with global parameters for the pathway model.

TO DO -----------------------------------------------------------------------------
- write this docstring
- load national data from file
- save vars to yml - what about dataframes? (repr?)
- only save inputs to Scenario() for init?
"""
import numpy as np
import pandas as pd
import os

from classes.units import Units
from classes.setup import Setup


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
        # Name that will also be used for output directory:
        self.name = 'scenario'

        # Which LSOAs will we use?
        self.mt_hub_postcodes = []
        self.limit_to_england = False
        self.limit_to_wales = False
        self.select_lsoa_method = 'nearest'
        self.region_type_for_lsoa_selection = None
        self.region_column_for_lsoa_selection = None

        # If stroke units
        self.limit_lsoa_to_regions = True


        self.run_duration = 365  # Days
        self.warm_up = 50

        # Which stroke team choice model will we use?
        self.destination_decision_type = 0
        #   0 is 'drip-and-ship',
        #   1 is 'mothership',
        #   2 is 'MSU'.

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
        #     'hospital_name3': {'Nearest_MT': 'EX25DW'},
        #     }
        self.services_updates = {}

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()
        # And create an output directory for this Scenario:
        dir_output = self.name
        # Return here because the output dir will be changed if
        # a dir with the same name already exists.
        dir_output = self.setup.create_output_dir(dir_output)

        # Convert run duration to minutes
        self.run_duration *= 1440

        if isinstance(self.region_type_for_lsoa_selection, str):
            # Load and parse hospital data
            dir_input = self.setup.dir_data
            file_input = self.setup.file_input_hospital_info
            path_to_file = os.path.join(dir_input, file_input)
            hospitals = pd.read_csv(path_to_file)
            # Regions to limit to:
            self.region_column_for_lsoa_selection = self._find_region_column(
                self.region_type_for_lsoa_selection,
                hospitals.columns
                )
        else:
            pass

        # # Load data:
        # # (run this after MT hospitals are updated in
        # # initial_data or kwargs).
        # self.load_data()

    # ###############################
    # ##### MAIN SETUP FUNCTION #####
    # ###############################
    def load_data(self):
        """
        TO DO - update me ------------------------------------------------------

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
        # Find which LSOAs are in these stroke teams' catchment areas:
        self._create_lsoa_travel_dicts()
        # Stores:
        # + self.lsoa_names
        #   --> saves to: file_selected_lsoas
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit
        # + self.lsoa_mt_travel_time
        # + self.lsoa_mt_unit
        # + self.lsoa_msu_travel_time
        # + self.lsoa_msu_unit

        # Import admissions statistics for those hospitals:
        self._load_admissions()
        # Stores:
        # + self.total_admissions
        # + self.lsoa_relative_frequency
        # + self.inter_arrival_time

    # ##########################
    # ##### AREAS TO MODEL #####   --> should this move to Units()?
    # ##########################
    def get_model_areas(self):
        try:
            df = self.selected_regions
        except AttributeError:
            # Load and parse area data
            dir_input = self.setup.dir_data
            file_input = self.setup.file_input_regions
            path_to_file = os.path.join(dir_input, file_input)
            df = pd.read_csv(path_to_file)

            # Add a "selected" column for user input.
            df['selected'] = 0
        return df

    def set_model_areas(self, df):
        # TO DO - run sanity checks

        # TO DO - add an option to load this from a custom file.
        self.selected_regions = df

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_regions
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file, index=False)

        try:
            df_units = self.unit_services
        except AttributeError:
            # Automatically create units based on these areas.
            df_units = self.get_unit_services()
            self.set_unit_services(df_units)

    # #########################
    # ##### UNIT SERVICES #####   --> should this move to Units()?
    # #########################
    def get_unit_services(self):
        try:
            df = self.unit_services
        except AttributeError:
            # Load and parse unit data
            dir_input = self.setup.dir_data
            file_input = self.setup.file_input_unit_services
            path_to_file = os.path.join(dir_input, file_input)
            df = pd.read_csv(path_to_file)

            # Drop the LSOA column.
            df = df.drop('LSOA11NM', axis='columns')
            # Add a "selected" column for user input.
            df['selected'] = 0

            try:
                # Load in the selected areas.
                df_areas = self.selected_regions
                # Which areas were selected?
                selected = df_areas['region'][df_areas['selected'] == 1]
                # Shouldn't contain repeats or NaN, but just in case:
                selected = selected.dropna().unique()
                # Set "selected" to 1 for any unit in the
                # selected areas.
                mask = df['region'].isin(selected)
                df.loc[mask, 'selected'] = 1
            except AttributeError:
                # self.selected_regions has not yet been set.
                pass
        return df

    def set_unit_services(self, df):
        # TO DO - run sanity checks

        try:
            df[['Easting', 'Northing', 'long_x', 'lat_x']]
        except KeyError:
            # Merge in geometry.
            # TO DO - just store this somewhere else:
            # - it's annoying when this runs twice accidentally and get loads of extra
            # geometry columns and suffixes. FIX ME! ----------------------------------------------
            # Load and parse geometry data
            dir_input = self.setup.dir_data
            file_input = self.setup.file_input_hospital_info
            path_to_file = os.path.join(dir_input, file_input)
            df_info = pd.read_csv(path_to_file)
            # Merge:
            df = pd.merge(
                df, df_info[['Postcode', 'Easting', 'Northing', 'long_x', 'lat_x']],
                left_on='Postcode', right_on='Postcode', how='left')

        # TO DO - add an option to load this from a custom file.
        self.unit_services = df

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_stroke_units
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file, index=False)

        # Calculate national unit info:
        self.load_units_data()

    def load_units_data(self):
        # ##### NATIONAL UNITS #####
        try:
            self.units
        except AttributeError:
            self.units = Units({'setup': self.setup})
        self.national_dict = self.units.load_data()

        # ##### SELECTED UNITS #####
        self._load_transfer_units()
        # WRITE ME TO DO -----------------------------------------------------------------

    def _load_transfer_units(self):

        # Merge in transfer unit names.
        # Load and parse hospital transfer data
        dir_input = self.setup.dir_output
        file_input = self.setup.file_national_transfer_units
        path_to_file = os.path.join(dir_input, file_input)
        transfer = pd.read_csv(path_to_file, index_col=0)

        dir_input = self.setup.dir_output
        file_input = self.setup.file_selected_stroke_units
        path_to_file = os.path.join(dir_input, file_input)
        hospitals = pd.read_csv(path_to_file, index_col=0)

        # Keep a copy of the coordinates:
        hospital_coords = hospitals.copy()
        hospital_coords = hospital_coords[[
            'Easting', 'Northing', 'long_x', 'lat_x']]  # TO DO - Fix this annoying suffix

        transfer = transfer.drop(['time_nearest_MT'], axis='columns')
        # Merge in the transfer unit coordinates:
        transfer = pd.merge(
            transfer, hospital_coords,
            left_on='name_nearest_MT', right_index=True,
            how='left', suffixes=('_mt', None)
            )

        transfer_hospitals = pd.merge(
            hospital_coords,
            transfer,
            left_index=True,
            right_index=True,
            how='left', suffixes=(None, '_mt')
        )
        # TO DO - tidy up the excess columns in hospitals ---------------------------

        # transfer_hospitals = transfer_hospitals.set_index('Postcode')
        self.transfer_hospitals = transfer_hospitals

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_transfer_units
        path_to_file = os.path.join(dir_output, file_name)
        transfer_hospitals.to_csv(path_to_file)

    # #########################
    # ##### SELECTED LSOA #####
    # #########################
    def find_lsoa_by_catchment(self):
        """
        TO DO - write me
        """
        df_results = self.units.find_lsoa_by_catchment(
            self.unit_services,
            self,
            treatment='IVT',
        )

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoa_by_catchment
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.lsoa_travel_by_catchment = df_results

    def find_lsoa_by_region_island(self):
        """
        TO DO - write me
        """
        df_results = self.units.find_lsoa_by_region_island(
            self.unit_services,
            self,
            treatment='IVT',
        )

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoa_by_region_island
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.lsoa_travel_by_region_island = df_results

    def set_lsoa_catchment_type(self, lsoa_catchment_type):
        """
        TO DO - write me
        """
        # TO DO - run sanity checks

        if lsoa_catchment_type == 'island':
            try:
                self.lsoa_names = self.lsoa_travel_by_region_island
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_by_region_island()
                self.lsoa_names = self.lsoa_travel_by_region_island
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_by_region_island)
        else:
            try:
                self.lsoa_names = self.lsoa_travel_by_catchment
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_by_catchment()
                self.lsoa_names = self.lsoa_travel_by_catchment
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_by_catchment)

        self.lsoa_catchment_type = lsoa_catchment_type

    def _create_lsoa_travel_dicts(self):
        """
        Convert LSOA travel time dataframe into separate dicts.
        """
        # Now create dictionaries of the LSOA travel times.
        df_travel = self.lsoa_names
        # TO DO - make this updateable for MT, MSU -------------------------------------
        # ..?
        treatments = ['IVT']

        # Separate out the columns and store in self:
        for treatment in treatments:
            travel_key = f'lsoa_{treatment}_travel_time'
            travel_val = df_travel[f'time_nearest_{treatment}']
            setattr(self, travel_key, travel_val)

            unit_key = f'lsoa_{treatment}_unit'
            unit_val = df_travel[f'postcode_nearest_{treatment}']
            setattr(self, unit_key, unit_val)

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
        dir_input = self.setup.dir_data
        file_input = self.setup.file_input_admissions
        path_to_file = os.path.join(dir_input, file_input)
        admissions = pd.read_csv(path_to_file)

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
