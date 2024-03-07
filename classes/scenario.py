"""
Scenario class with global parameters for the pathway model.

TO DO -----------------------------------------------------------------------------
- write this docstring
- only save inputs to Scenario() for init?

        Assume that if this is run directly instead of through
        find_lsoa_catchment_nearest that the other dataframes have
        already been updated - have columns for:
        + contains_selected_lsoa,
        + contains_unit_catching_lsoa,
        + catches_lsoa_in_selected_region

        # TO DO - need to run the checks for LSOA catchment in other regions, by other units
        # even when they've been imported from file.

Usually the scenario.yml file should contain:

'name'  # this scenario
# Geography setup: for LSOA, region, units...
'limit_to_england',  # bool
'limit_to_wales',  # bool
'lsoa_catchment_type',  # 'island' or 'nearest'


TO DO - should this all be functions too? Not a class?
No, too many files and stuff in self.
"""
import numpy as np
import pandas as pd
import os
import yaml

# from classes.calculations import Calculations


class Scenario(object):
    """
    Global variables for model.

    class Scenario():

    Attributes
    ----------

    Methods
    -------

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
        # Load existing data from this dir:
        self.load_dir = None

        # ----- Geography -----
        # Which LSOAs, stroke units, and regions will we use?
        self.limit_to_england = False
        self.limit_to_wales = False
        self.lsoa_catchment_type = 'nearest'

        # ----- Overwrite default values -----
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # ----- Load helper classes -----
        # Create calculations object:
        # self.calculations = Calculations({'setup': self.setup})

        # ----- Load Scenario from files -----
        if self.load_dir is None:
            pass
        else:
            self.load_scenario_from_files()

    def load_scenario_from_files(self, path_to_scenario_yml):
        """
        """
        # Import the kwargs from provided yml file:
        with open(path_to_scenario_yml, 'r') as f:
            scenario_vars_imported = yaml.safe_load(f)
        # Save the imported kwargs to self:
        for key, val in scenario_vars_imported.items():
            setattr(self, key, val)

    def save_to_file(self, file_setup_vars='scenario_output.yml'):
        """Save the variable dict as a .yml file."""
        scenario_vars = vars(self)

        # Only keep a selection of params:
        types_to_keep = [float, int, str]

        vars_to_save = {}
        for key, val in scenario_vars.items():
            if any([isinstance(val, t) for t in types_to_keep]):
                if isinstance(val, np.float64):
                    val = val.tolist()
                vars_to_save[key] = val

        with open(file_setup_vars, 'w') as f:
            yaml.dump(vars_to_save, f)

    # ###############################
    # ##### MAIN SETUP FUNCTION #####
    # ###############################
    def process_scenario(self, units):
        """
        Some of the attributes might already exist depending on how
        the data has been loaded in, so at each step check if it
        already exists or not.
        """
        self.df_selected_units = units

        self.get_transfer_units()

        self.calculate_lsoa_catchment()

        admissions = self.load_admissions()

    # #########################
    # ##### UNIT SERVICES #####
    # #########################
    def get_unit_services(self):
        """
        write me
        """
        # TO DO - replace with relative import
        # Load and parse unit data
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_unit_services)
        df = pd.read_csv(path_to_file)

        # Drop the LSOA column.
        df = df.drop('lsoa', axis='columns')
        # Add a "selected" column for user input.
        df['selected'] = 0

        return df

    # ##########################
    # ##### TRANSFER UNITS #####
    # ##########################
    def get_transfer_units(self):
        """
        write me

        If exists, don't load?
        """
        # Find which IVT units are feeders to each MT unit:
        transfer = self.find_national_mt_feeder_units()
        transfer = transfer.rename(columns={'from_postcode': 'postcode'})
        # transfer = transfer.drop(['time_nearest_mt'], axis='columns')
        # Index: 'Postcode'
        # Columns: 'name_nearest_mt'

        units = self.df_selected_units
        # Index: 'Postcode'
        # Columns: names, services, regions etc. ...

        # Label selected stroke units.
        selected_units = units['postcode'][units['selected'] == 1]
        mask = transfer['postcode'].isin(selected_units)
        transfer['selected'] = 0
        transfer['selected'][mask] = 1
        transfer = transfer.set_index('postcode')

        self.transfer = transfer

    def find_national_mt_feeder_units(self, df_stroke_teams):
        """
        Find catchment areas for national hospitals offering MT.

        For each stroke unit, find the name of and travel time to
        its nearest MT unit. Wheel-and-spoke model. If the unit
        is an MT unit then the travel time is zero.

        Stores
        ------

        national_ivt_feeder_units:
            pd.DataFrame. Each row is a stroke unit. Columns are
            its postcode, the postcode of the nearest MT unit,
            and travel time to that MT unit.
        """
        # Get list of services that each stroke team provides:
        # df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'postcode', 'SSNAP name', 'use_ivt', 'use_mt', 'use_msu'
        # where the "use_" columns contain 0 (False) or 1 (True).

        # Pick out the names of hospitals offering IVT:
        mask_ivt = (df_stroke_teams['use_ivt'] == 1)
        ivt_hospital_names = df_stroke_teams['postcode'][mask_ivt].values
        # Pick out the names of hospitals offering MT:
        mask_mt = (df_stroke_teams['use_mt'] == 1)
        mt_hospital_names = df_stroke_teams['postcode'][mask_mt].values

        # TO DO - change to relative import
        # Firstly, determine MT feeder units based on travel time.
        # Each stroke unit will be assigned the MT unit that it is
        # closest to in travel time.
        # Travel time matrix between hospitals:
        path_to_file = os.path.join(
            self.setup.dir_reference_data,
            self.setup.file_input_travel_times_inter_unit
            )
        df_time_inter_hospital = pd.read_csv(path_to_file,
                                             index_col='from_postcode')
        # Reduce columns of inter-hospital time matrix to just MT hospitals:
        df_time_inter_hospital = df_time_inter_hospital[mt_hospital_names]

        # From this reduced dataframe, pick out
        # the smallest time in each row and
        # the MT hospital that it belongs to.
        # Store the results in this DataFrame:
        df_nearest_mt = pd.DataFrame(index=df_time_inter_hospital.index)
        # The smallest time in each row:
        df_nearest_mt['transfer_unit_travel_time'] = (
            df_time_inter_hospital.min(axis='columns'))
        # The name of the column containing the smallest time in each row:
        df_nearest_mt['transfer_unit_postcode'] = (
            df_time_inter_hospital.idxmin(axis='columns'))

        # Only define transfer for units offering IVT.
        mask = df_nearest_mt.index.isin(ivt_hospital_names)
        df_nearest_mt.loc[~mask, 'transfer_unit_postcode'] = 'none'

        # Update the feeder units list with anything specified
        # by the user.
        df_services_to_update = df_stroke_teams[
            df_stroke_teams['transfer_unit_postcode'] != 'nearest']
        units_to_update = df_services_to_update['postcode'].values
        transfer_units_to_update = df_services_to_update[
            'transfer_unit_postcode'].values
        for u, unit in units_to_update:
            transfer_unit = transfer_units_to_update[u]
            if transfer_unit == 'none':
                # Set values to missing:
                transfer_unit = pd.NA
                mt_time = pd.NA
            else:
                # Find the time to this MT unit.
                mt_time = df_time_inter_hospital.loc[unit][transfer_unit]

            # Update the chosen nearest MT unit name and time.
            df_nearest_mt.at[unit, 'transfer_unit_postcode'] = transfer_unit
            df_nearest_mt.at[unit, 'transfer_unit_travel_time'] = mt_time

        return df_nearest_mt

    # #########################
    # ##### SELECTED LSOA #####
    # #########################
    def calculate_lsoa_catchment(self):
        """
        TO DO - write me
        """
        units = self.df_selected_units
        regions_selected = sorted(list(set(units['region_code'])))

        if self.lsoa_catchment_type == 'island':
            # Only use the selected stroke units:
            teams_to_limit = units['postcode'][units['selected'] == 1]
            # Find list of selected regions:
            regions_to_limit = regions_selected
        else:
            teams_to_limit = []
            regions_to_limit = []

        # For all LSOA:
        df_catchment = self.find_lsoa_catchment(teams_to_limit)

        # Mark selected LSOA:
        df_catchment = self.limit_lsoa_catchment_to_selected_units(
            df_catchment,
            regions_selected,
            regions_to_limit=regions_to_limit,
            limit_to_england=self.limit_to_england,
            limit_to_wales=self.limit_to_wales
            )

    def find_each_lsoa_chosen_unit(self, df_time_lsoa_to_units):
        """

        """
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_to_units.index)
        # The smallest time in each row:
        df_results['time_nearest'] = (
            df_time_lsoa_to_units.min(axis='columns'))
        # The name of the column containing the smallest
        # time in each row:
        df_results['unit_postcode'] = (
            df_time_lsoa_to_units.idxmin(axis='columns'))
        return df_results

    def find_lsoa_catchment(
            self,
            teams_to_limit=[]
            ):
        # TO DO - change to relative import
        # Load travel time matrix:
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_travel_times)
        df_time_lsoa_to_units = pd.read_csv(path_to_file, index_col='LSOA')
        # Each column is a postcode of a stroke team and
        # each row is an LSOA name (LSOA11NM).

        # Limit columns to requested units:
        if len(teams_to_limit) > 0:
            df_time_lsoa_to_units = df_time_lsoa_to_units[teams_to_limit]

        # Assign LSOA by catchment area of these stroke units.
        df_catchment = self.find_each_lsoa_chosen_unit(
            df_time_lsoa_to_units)
        return df_catchment

    def limit_lsoa_catchment_to_selected_units(
            self,
            df_catchment,
            regions_selected,
            regions_to_limit=[],
            limit_to_england=False,
            limit_to_wales=False
            ):
        # TO DO - change to relative import
        # Load in all LSOA names, codes, regions...
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_lsoa_regions)
        df_lsoa = pd.read_csv(path_to_file)
        # Columns: [lsoa, lsoa_code, region_code, region, region_type,
        #           icb_code, icb, isdn]

        # Keep a copy of the original catchment columns for later:
        cols_df_catchment = df_catchment.columns
        # Merge in region information to catchment:
        df_catchment.reset_index(inplace=True)
        df_catchment = pd.merge(
            df_catchment, df_lsoa,
            left_on='LSOA', right_on='lsoa', how='left'
        )
        df_catchment.drop('lsoa', axis='columns', inplace=True)
        df_catchment.set_index('LSOA', inplace=True)

        # Limit rows to LSOA in requested regions:
        if len(regions_to_limit) > 0:
            mask = df_catchment['region_code'].isin(regions_to_limit)
            df_catchment = df_catchment.loc[mask].copy()
        else:
            # Find where the results data is in selected regions:
            mask = df_catchment['region_code'].isin(regions_selected)
            # Find list of units catching any LSOA in selected regions:
            units_catching_lsoa = sorted(list(set(
                df_catchment.loc[mask]['unit_postcode'])))

            # Limit the results to only
            # LSOAs that are caught by units
            # that catch any LSOA in the selected regions.
            mask = df_catchment[
                'unit_postcode'].isin(units_catching_lsoa)

        df_catchment['selected'] = 0
        df_catchment['selected'][mask] = 1

        # If requested, remove England or Wales.
        if limit_to_england:
            mask = df_lsoa['region_type'] == 'LHB'
            df_catchment['selected'][mask] = 0
        elif limit_to_wales:
            mask = df_lsoa['region_type'] == 'SICBL'
            df_catchment['selected'][mask] = 0

        # Restore the shortened catchment DataFrame to its starting columns
        # plus the useful regions:
        cols = cols_df_catchment + ['selected']
        # ['region', 'region_code', 'region_type']
        df_catchment = df_catchment[cols]

        return df_catchment

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
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_admissions)
        admissions = pd.read_csv(path_to_file)
        return admissions

    def match_admissions_to_selected_lsoa(self, admissions):
        """
        write me
        """
        # Keep only these LSOAs in the admissions data:
        admissions = pd.merge(left=self.df_lsoa, right=admissions,
                              left_on='lsoa', right_on='area', how='left')

        admissions_mask = admissions.loc[admissions['selected'] == 1].copy()

        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        total_admissions = np.round(
            admissions_mask["admissions"].sum(), 0).tolist()

        # Relative frequency of admissions across a year:
        admissions_mask['relative_frequency'] = (
            admissions_mask['admissions'] / total_admissions)

        # Merge this info back into the main DataFrame:
        admissions = pd.merge(
            admissions, admissions_mask, on='lsoa', how='left')

        # Set index to both LSOA name and code so that both follow
        # through to all of the results data.
        admissions = admissions.set_index(['area', 'lsoa_code'])
        return admissions
