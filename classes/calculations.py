"""
Units class for defining choice of stroke unit through the pathway.
"""
import pandas as pd
import os  # For checking directory existence

from classes.setup import Setup


class Calculations(object):
    """
    Links between stroke units.
    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

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

        # Load data:
        # (run this after MT hospitals are updated in
        # initial_data or kwargs).
        # self.load_data()

    def load_data(self):
        """
        Load required data.

        Stores the following in the Globvars object:
        + national_hospital_services
        + national_lsoa_nearest_units
        + national_ivt_feeder_units
        + national_mt_transfer_time
        + national_mt_transfer_unit

        More details on each attribute are given in the docstrings
        of the methods that create them.
        """
        # ##### NATIONAL INFORMATION #####
        # Load information about all stroke units nationally,
        # whether they're being highlighted in the simulation or not.
        # Find which stroke units provide IVT, MT, and MSU:
        self._set_national_hospital_services()
        # Stores:
        # + self.national_hospital_services
        #   --> saves to: file_national_unit_services

        # Find each LSOA's nearest IVT, MT, and MSU units:
        # self._find_national_lsoa_nearest_units()  # TEMP COMMENTED OUT - FIX ME PLEASE - TO DO
        # Stores:
        # + self.national_lsoa_nearest_units
        #   --> saves to: file_national_lsoa_travel

        # Find which IVT units are feeders to each MT unit:
        self._find_national_mt_feeder_units()
        # Stores:
        # + self.national_ivt_feeder_units
        #   --> saves to: file_national_transfer_units

        # Transfer stroke unit data.
        self._find_national_transfer_travel()
        # Stores:
        # + self.national_mt_transfer_time
        # + self.national_mt_transfer_unit
        #   --> no file saved.

        # Place everything useful into a dict for returning:
        national_dict = dict(
            hospital_services=self.national_hospital_services,
            # lsoa_nearest_units=self.national_lsoa_nearest_units,
            ivt_feeder_units=self.national_ivt_feeder_units,
            mt_transfer_time=self.national_mt_transfer_time,
            mt_transfer_unit=self.national_mt_transfer_unit,
        )

        return national_dict

    # ################################
    # ##### NATIONAL INFORMATION #####
    # ################################
    def _set_national_hospital_services(self):
        """
        Make table of which stroke units provide which treatments.

        Each stroke unit has a flag in this table for each of:
        + use_ivt
        + use_mt
        + use_msu
        The value is set to either 0 (not provided) or 1 (provided).

        Most of the values are stored in a reference file but
        they can be updated by the user with the ... UPDATE ME ------------------

        These values should be set for all units nationally,
        because otherwise patients from e.g. Newcastle will have their
        nearest stroke unit set to e.g. Cornwall.

        Stores
        ------

        national_hospital_services:
            pd.DataFrame. Each stroke team's services provided.
            Columns for whether a team provides IVT, MT, and MSU.
        """
        # Load default stroke unit services:
        dir_input = self.setup.dir_output
        file_input = self.setup.file_selected_stroke_units
        path_to_file = os.path.join(dir_input, file_input)
        services = pd.read_csv(path_to_file)
        # Each row is a stroke unit. The columns are 'postcode' and
        # 'SSNAP name' (str), and 'use_ivt', 'use_mt', and 'use_msu'
        # (int | bool).

        # Store national hospitals and their services in self.
        self.national_hospital_services = services

        # # Save output to output folder.
        # dir_output = self.setup.dir_output
        # file_name = self.setup.file_national_unit_services
        # path_to_file = os.path.join(dir_output, file_name)
        # services.to_csv(path_to_file, index=False)

    def find_lsoa_catchment_nearest(
            self,
            df_units,
            scenario,
            treatment='IVT'
            ):
        """
        TO DO - write me -----------------------------------------------------------
        """
        # Find list of stroke units catching these LSOA.
        # Limit to units offering IVT:
        df_units = df_units[df_units[f'use_{treatment}'] == 1]
        # List of teams to use:
        teams = df_units['postcode'].values

        # Load travel time matrix:
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_travel_times
        path_to_file = os.path.join(dir_input, file_input)
        df_time_lsoa_hospital = pd.read_csv(
            path_to_file,
            index_col='LSOA'
            )
        # Each column is a postcode of a stroke team and
        # each row is an LSOA name (LSOA11NM).

        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_hospital.index)
        # The smallest time in each row:
        df_results[f'time_nearest_{treatment}'] = (
            df_time_lsoa_hospital[teams].min(axis='columns'))
        # The name of the column containing the smallest
        # time in each row:
        df_results[f'postcode_nearest_{treatment}'] = (
            df_time_lsoa_hospital[teams].idxmin(axis='columns'))

        # Load in all LSOA names, codes, regions...
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_lsoa_regions
        path_to_file = os.path.join(dir_input, file_input)
        df_lsoa = pd.read_csv(path_to_file)
        # Full list of columns:
        # [LSOA11NM, LSOA11CD, region_code, region, region_type,
        #  ICB22CD, ICB22NM, ISDN]

        # If requested, limit to England or Wales.
        if scenario.limit_to_england:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'SICBL']
        elif scenario.limit_to_wales:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'LHB']

        # Find selected regions:
        # Columns [region, region_code, region_type,
        #          ICB22CD, ICB22NM, ISDN, selected]
        df_regions = scenario.selected_regions
        region_list = sorted(list(set(
            df_regions['region_code'][df_regions['selected'] == 1])))
        # Find all LSOA within selected regions.
        df_lsoa_in_regions = df_lsoa[df_lsoa['region_code'].isin(region_list)].copy()
        # Find list of units catching any LSOA in selected regions.
        mask = df_results.index.isin(df_lsoa_in_regions['lsoa'])
        df_results_in_regions = df_results.loc[mask]
        units_catching_lsoa = list(set(
            df_results_in_regions[f'postcode_nearest_{treatment}']))
        # Find regions containing these units:
        mask = df_units['postcode'].isin(units_catching_lsoa)
        region_codes_containing_units = list(set(
            df_units.loc[mask, 'region_code']))

        # Separate out LSOA caught by selected units.
        # Limit units to those offering IVT:
        df_units = df_units[df_units[f'use_{treatment}'] == 1]
        selected_units = df_units['postcode'][df_units['selected']  == 1]
        mask = df_results[f'postcode_nearest_{treatment}'].isin(selected_units)
        df_results = df_results.loc[mask]

        # Limit to just these LSOA:
        df_results = df_results.reset_index()
        df_results = pd.merge(
            df_results, df_lsoa[['lsoa', 'lsoa_code', 'region_code']],
            left_on='LSOA', right_on='lsoa', how='inner'
            )

        # Find regions containing LSOA:
        region_codes_containing_lsoa = list(set(
            df_results['region_code']))

        df_results = df_results.drop(['LSOA', 'region_code'], axis='columns')
        df_results = df_results.set_index('lsoa')

        # Reorder columns:
        df_results = df_results[[
            'lsoa_code',
            f'postcode_nearest_{treatment}',
            f'time_nearest_{treatment}'
            ]]

        return (df_results, region_codes_containing_lsoa,
                region_codes_containing_units, units_catching_lsoa)

    def find_catching_regions_and_units():
        """
        Find these things:

        region_codes_containing_lsoa,
        region_codes_containing_units,
        units_catching_lsoa
        """


    def find_lsoa_catchment_island(
            self,
            df_units,
            scenario,
            treatment='IVT',
            ):
        """
        TO DO - write me ----------------------------------------------------------
        """
        # Load travel time matrix:
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_travel_times
        path_to_file = os.path.join(dir_input, file_input)
        df_time_lsoa_hospital = pd.read_csv(
            path_to_file,
            index_col='LSOA'
            )
        # Each column is a postcode of a stroke team and
        # each row is an LSOA name (LSOA11NM).

        # Limit units to those offering IVT:
        df_units = df_units[df_units[f'use_{treatment}'] == 1]
        # Limit to selected units:
        df_units = df_units[df_units['selected'] == 1]
        # List of teams to use:
        teams = df_units['postcode'].values
        # Limit the travel time list to only selected units.
        df_time_lsoa_hospital = df_time_lsoa_hospital[teams]

        # Assign LSOA by catchment area of these stroke units.
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_hospital.index)
        # The smallest time in each row:
        df_results[f'time_nearest_{treatment}'] = (
            df_time_lsoa_hospital[teams].min(axis='columns'))
        # The name of the column containing the smallest
        # time in each row:
        df_results[f'postcode_nearest_{treatment}'] = (
            df_time_lsoa_hospital[teams].idxmin(axis='columns'))

        # Load in all LSOA names, codes, regions...
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_lsoa_regions
        path_to_file = os.path.join(dir_input, file_input)
        df_lsoa = pd.read_csv(path_to_file)
        # Full list of columns:
        # [LSOA11NM, LSOA11CD, region_code, region, region_type,
        #  ICB22CD, ICB22NM, ISDN]
        # Only keep LSOA name and code and region name and code:
        cols_to_keep = [
            'lsoa', 'lsoa_code', 'region', 'region_code', 'region_type']
        df_lsoa = df_lsoa[cols_to_keep]

        # If requested, limit to England or Wales.
        if scenario.limit_to_england:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'SICBL']
        elif scenario.limit_to_wales:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'LHB']

        # Find selected regions:
        # Columns [region, region_code, region_type,
        #          ICB22CD, ICB22NM, ISDN, selected]
        df_regions = scenario.selected_regions
        region_list = sorted(list(set(
            df_regions['region_code'][df_regions['selected'] == 1])))

        # Find all LSOA within selected regions.
        df_lsoa_in_regions = df_lsoa[df_lsoa['region_code'].isin(region_list)]

        # Limit the results list to only selected LSOA.
        df_results = df_results.reset_index()
        df_results = pd.merge(
            df_results, df_lsoa_in_regions[['lsoa', 'lsoa_code']],
            left_on='LSOA', right_on='lsoa', how='right'
            )
        df_results = df_results.drop('LSOA', axis='columns')
        df_results = df_results.set_index('lsoa')

        # Reorder columns:
        df_results = df_results[[
            'lsoa_code',
            f'postcode_nearest_{treatment}',
            f'time_nearest_{treatment}'
            ]]
        return df_results

    def _find_national_mt_feeder_units(self):
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
        df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'postcode', 'SSNAP name', 'use_ivt', 'use_mt', 'use_msu'
        # where the "use_" columns contain 0 (False) or 1 (True).

        # Pick out the names of hospitals offering IVT:
        mask_ivt = (df_stroke_teams['use_ivt'] == 1)
        ivt_hospital_names = df_stroke_teams['postcode'][mask_ivt].values
        # Pick out the names of hospitals offering MT:
        mask_mt = (df_stroke_teams['use_mt'] == 1)
        mt_hospital_names = df_stroke_teams['postcode'][mask_mt].values

        # Firstly, determine MT feeder units based on travel time.
        # Each stroke unit will be assigned the MT unit that it is
        # closest to in travel time.
        # Travel time matrix between hospitals:
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_travel_times_inter_unit
        path_to_file = os.path.join(dir_input, file_input)
        df_time_inter_hospital = pd.read_csv(
            path_to_file,
            index_col='from_postcode'
            )
        # Reduce columns of inter-hospital time matrix to just MT hospitals:
        df_time_inter_hospital = df_time_inter_hospital[mt_hospital_names]

        # From this reduced dataframe, pick out
        # the smallest time in each row and
        # the MT hospital that it belongs to.
        # Store the results in this DataFrame:
        df_nearest_mt = pd.DataFrame(index=df_time_inter_hospital.index)
        # The smallest time in each row:
        df_nearest_mt['time_nearest_mt'] = (
            df_time_inter_hospital.min(axis='columns'))
        # The name of the column containing the smallest time in each row:
        df_nearest_mt['name_nearest_mt'] = (
            df_time_inter_hospital.idxmin(axis='columns'))

        # Update the feeder units list with anything specified
        # by the user.
        df_services = self.national_hospital_services
        df_services_to_update = df_services[
            df_services['transfer_unit_postcode'] != 'nearest']
        units_to_update = df_services_to_update['postcode'].values
        transfer_units_to_update = df_services_to_update['transfer_unit_postcode'].values
        for u, unit in units_to_update:
            transfer_unit = transfer_units_to_update[u]

            # Find the time to this MT unit.
            mt_time = df_time_inter_hospital.loc[unit][transfer_unit]

            # Update the chosen nearest MT unit name and time.
            df_nearest_mt.at[unit, 'name_nearest_mt'] = transfer_unit
            df_nearest_mt.at[unit, 'time_nearest_mt'] = mt_time

        # Only keep units offering IVT.
        mask = df_nearest_mt.index.isin(ivt_hospital_names)
        df_nearest_mt = df_nearest_mt[mask]

        # Store in self:
        self.national_ivt_feeder_units = df_nearest_mt

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_national_transfer_units
        path_to_file = os.path.join(dir_output, file_name)
        df_nearest_mt.to_csv(path_to_file)

    def _find_national_transfer_travel(self):
        """
        Data for the transfer stroke unit of each national stroke unit.

        Stores
        ------

        national_mt_transfer_time:
            dict. Each stroke unit's travel time to their nearest
            MT transfer unit.

        national_mt_transfer_unit:
            dict. Each stroke unit's nearest MT transfer unit's name.
        """
        # Load and parse inter hospital travel time for MT
        inter_hospital_time = self.national_ivt_feeder_units

        self.national_mt_transfer_time = dict(
            inter_hospital_time['time_nearest_mt'])
        self.national_mt_transfer_unit = dict(
            inter_hospital_time['name_nearest_mt'])
