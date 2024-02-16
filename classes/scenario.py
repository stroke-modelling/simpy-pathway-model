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
        self.limit_to_england = True
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

    # ##########################
    # ##### ... write me TO DO #####
    # ##########################
    def _find_region_column(self, region_type, columns):
        """
        Find the column name that best matches the region type.
        """
        if region_type in columns:
            # Use this column.
            col = region_type
        else:
            # Guess which one is intended.
            cols = [c for c in columns if region_type in c]
            # Prioritise the ones that start with the region type.
            cols_prefix = [c for c in cols if (
                (len(c) >= len(region_type)) &
                (c[:len(region_type)] == region_type)
            )]
            # Prioritise the ones that end with 'NM':
            cols_suffix = [c for c in cols if c[-2:] == 'NM']
            if len(cols_suffix) > 0:
                col = cols_suffix[0]
            elif len(cols_prefix) > 0:
                col = cols_prefix[0]
            elif len(cols) > 0:
                col = cols[0]
            else:
                # This shouldn't happen.
                col = columns[0]
                # TO DO - raise an exception or something here. -----------------------------
        return col

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
        # Find which LSOAs are in these stroke teams' catchment areas:
        self._load_lsoa_names()
        # Stores:
        # + self.lsoa_names
        #   --> saves to: file_selected_lsoas
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit
        # + self.lsoa_mt_travel_time
        # + self.lsoa_mt_unit
        # + self.lsoa_msu_travel_time
        # + self.lsoa_msu_unit

        # Find which regions contain an LSOA.
        self._link_lsoa_to_regions()
        # --> saves to: file_selected_regions

        # Import admissions statistics for those hospitals:
        self._load_admissions()
        # Stores:
        # + self.total_admissions
        # + self.lsoa_relative_frequency
        # + self.inter_arrival_time

    # ##########################
    # ##### SELECTED UNITS #####
    # ##########################
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
    def _load_lsoa_names(self):
        """
        TO DO - make this simpler!! -----------------------------------------------

        WIP
        Load names of LSOAs in catchment areas of chosen stroke teams.

        Stores
        ------
        lsoa_names:
            np.array. Names of all LSOAs considered.

        lsoa_ivt_travel_time:
            dict. Each LSOA's nearest IVT unit travel time.

        lsoa_ivt_unit:
            dict. Each LSOA's nearest IVT unit name.

        lsoa_mt_travel_time:
            dict. Each LSOA's nearest MT unit travel time.

        lsoa_mt_unit:
            dict. Each LSOA's nearest MT unit name.
        """
        # Load data on LSOA names, codes, regions...
        dir_input = self.setup.dir_data
        file_input = self.setup.file_input_lsoa_regions
        path_to_file = os.path.join(dir_input, file_input)
        df_regions = pd.read_csv(path_to_file)
        # Only keep LSOA name, code, and coordinates:
        cols_to_keep = [
            'LSOA11NM', 'LSOA11CD',
            # 'LSOA11BNG_N', 'LSOA11BNG_E',
            # 'LSOA11LONG', 'LSOA11LAT',
            self.region_column_for_lsoa_selection
        ]
        df_regions = df_regions[cols_to_keep]

        # Limit the available LSOAs if required.
        if len(self.mt_hub_postcodes) > 0:
            if self.select_lsoa_method == 'nearest':
                # Only include LSOAs that have their nearest
                # {stroke unit type} in the selected units.
                # The list of LSOAs will be different for the
                # same list of selected stroke units depending
                # on whether the model type is drip-and-ship or
                # mothership. Some units have their nearest MT
                # unit in the selected list but their transfer
                # MT unit outside the selected list.
                lsoas_to_include = self._select_lsoas_by_nearest()
            else:
                lsoas_to_include = self._select_lsoas_by_region()
        elif self.limit_to_england:
            # Limit the data to English LSOAs only.
            # The LSOA11CD (ONS code for each LSOA) begins with
            # an "E" for English and "W" for Welsh LSOAs.
            # All other characters are numbers.
            mask_england = df_regions['LSOA11CD'].str.startswith('E')
            lsoas_to_include = df_regions['LSOA11NM'][mask_england]
        else:
            # Just use all LSOAs in the file.
            lsoas_to_include = df_regions['LSOA11NM']

        # Reduce the full LSOA data to just these chosen LSOAs:
        df_regions = pd.merge(
            df_regions, lsoas_to_include,
            left_on='LSOA11NM', right_on='LSOA11NM',
            how='right'
            )

        # TO DO - can the travel stuff move elsewhere? ---------------------------------

        # Stroke unit data for each LSOA.
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
            df_regions,
            df_travel.drop(['LSOA11CD'], axis='columns'),
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

        # Limit to just the units for the selected model type.
        s = ('IVT' if self.destination_decision_type == 0
             else 'MT' if self.destination_decision_type == 1
             else 'MSU')
        cols_to_drop = []
        for d in ['IVT', 'MT', 'MSU']:
            cols = [
                    f'postcode_nearest_{d}',
                    f'time_nearest_{d}',
                    f'ssnap_name_nearest_{d}',
                ]
            if d != s:
                cols_to_drop += cols
            else:
                cols_to_drop += cols[1:]
                cols = cols[:1]
                cols_to_rename = dict(zip(
                    cols, [c.split(f'_{s}')[0] for c in cols]))
        df_regions = df_travel
        df_regions = df_regions.drop(cols_to_drop, axis='columns')
        df_regions = df_regions.rename(columns=cols_to_rename)

        # Store in self:
        self.lsoa_names = df_regions

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoas
        path_to_file = os.path.join(dir_output, file_name)
        df_regions.to_csv(path_to_file)#, index=False)

    def _find_lsoa_catchment_mask(self, df_travel, col):
        # Which LSOAs are in the catchment areas for these units?
        # For each stroke team, make a long list of True/False for
        # whether each LSOA has this as its nearest unit.
        # Assume that "hospitals" has "Postcode" as its index.
        lsoa_bool = [df_travel[col].str.contains(s)
                    for s in self.hospitals.index.values]
        # Mask is True for any LSOA that is True in any of the
        # lists in lsoa_bool.
        mask = np.any(lsoa_bool, axis=0)
        return mask

    def _select_lsoas_by_nearest(self):
        """
        Limit LSOAs to those whose nearest stroke units are selected.
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

        # Limit to just the LSOAs for the selected model type.
        if self.destination_decision_type == 0:
            # Drip and ship.
            col = 'postcode_nearest_IVT'
        elif self.destination_decision_type == 1:
            # Mothership.
            col = 'postcode_nearest_MT'
        else:
            # MSU / other.
            col = 'postcode_nearest_MSU'

        mask = self._find_lsoa_catchment_mask(df_travel, col)

        # Limit the data to just these LSOAs:
        lsoas_to_include = df_travel['LSOA11NM'][mask]

        return lsoas_to_include

    def _select_lsoas_by_region(self):
        """
        Limit LSOAs to those in the same region as stroke units.

        TO DO - new function - limit big LSOA list to only those in the requested regions.
        And set all other stroke unit services to zero?
        """
        # Column:
        col = self.region_column_for_lsoa_selection
        # List of hospitals selected and the regions containing them:
        hospitals = self.hospitals

        # Pick out the region names with repeats:
        regions = hospitals[col].copy()
        # Remove missing values:
        regions = regions.dropna()
        # Remove repeats:
        regions = list(set(regions))

        # Load data on LSOA names, codes, regions...
        dir_input = self.setup.dir_data
        file_input = self.setup.file_input_lsoa_regions
        path_to_file = os.path.join(dir_input, file_input)
        df_regions = pd.read_csv(path_to_file)
        # Each row is a different LSOA and the columns include
        # LSOA11NM, LSOA11CD, longitude and latitude, and larger
        # regional groupings (e.g. Clinical Care Group names).

        # Which LSOAs are in the selected regions?
        # For each region, make a long list of True/False for
        # whether each LSOA is in this region.
        # List of LSOAs:
        lsoa_bool = [
            df_regions[col].str.contains(s)
            for s in regions
            ]
        # Sometimes get missing values, not True or False,
        # e.g. when comparing Welsh LSOAs with England-only region types.
        # Change any missing values to False:
        lsoa_bool = [s.fillna(False) for s in lsoa_bool]
        # Mask is True for any LSOA that is True in any of the
        # lists in lsoa_bool.
        mask = np.any(lsoa_bool, axis=0)
        # Limit the data to just these LSOAs:
        lsoas_to_include = df_regions['LSOA11NM'][mask]

        # TO DO ---------------------------------------------------------------------------
        # Need to make sure that LSOAs within the region but closer
        # to a stroke unit outside the region are forced to travel to
        # a stroke unit within the region.
        # Maybe just set all other stroke unit services to 0?
        return lsoas_to_include

    def _link_lsoa_to_regions(self):
        """
        write me
        new file, link LSOA names to regions containing them
        - just do a subset of the input file and save it

        # TO DO - new regions file. region / type / contains selected unit / LSOA
        # so that can mask by type column to get a list of regions used.
        # Also save this to file.
        """
        # Load data on LSOA names, codes, regions...
        dir_input = self.setup.dir_data
        file_input = self.setup.file_input_lsoa_regions
        path_to_file = os.path.join(dir_input, file_input)
        df_lsoa_regions = pd.read_csv(path_to_file)

        # TO DO - pick out which LSOA will be included in the model. Two model types:
        # either limit to the exact boundaries of the selected regions,
        # or use the catchment areas of all stroke units that were selected.

        # Save to self:
        self.region_choice = df

        # Save to file:
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_regions
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file)#, index=False)

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
