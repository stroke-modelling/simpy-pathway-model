"""
Combine features from multiple runs.

TO DO - write me --------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import os
from itertools import combinations

from classes.setup import Setup


class Combine(object):
    """
    Combine files from multiple runs of the pathway.

    class Combine():

    TO DO - write me
    """
    def __init__(self, *initial_data, **kwargs):

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

    def combine_files(self):
        self.combine_selected_lsoa()
        self.combine_selected_regions()
        self.combine_results_summary_by_lsoa()
        self.combine_results_summary_by_admitting_unit()

    # ##########################
    # ##### SPECIFIC FILES #####
    # ##########################

    def combine_selected_lsoa(self):
        """Combine selected LSOA"""
        file_to_merge = self.setup.file_selected_lsoas

        df = self._merge_multiple_dataframes(file_to_merge)

        # Save to file:
        output_dir = self.setup.dir_output_all_runs
        output_filename = self.setup.file_combined_selected_lsoas
        path_to_file = os.path.join(output_dir, output_filename)
        df.to_csv(path_to_file, index=False)

    def combine_selected_regions(self):
        """Combine selected regions"""
        file_to_merge = self.setup.file_selected_regions

        df = self._hstack_multiple_dataframes(file_to_merge)

        # Save to file:
        output_dir = self.setup.dir_output_all_runs
        output_filename = self.setup.file_combined_selected_regions
        path_to_file = os.path.join(output_dir, output_filename)
        df.to_csv(path_to_file)

    def combine_results_summary_by_lsoa(self):
        """Group by LSOA summary"""
        file_to_merge = self.setup.file_results_summary_by_lsoa

        data = self._stack_multiple_dataframes(file_to_merge)

        col_to_group = data.columns[0]
        cols_to_keep = ['utility_shift_mean', 'mRS shift_mean', 'mRS 0-2_mean']
        df = self._group_data_and_diff(data, col_to_group, cols_to_keep)

        # Save to file:
        output_dir = self.setup.dir_output_all_runs
        output_filename = self.setup.file_combined_results_summary_by_lsoa
        path_to_file = os.path.join(output_dir, output_filename)
        df.to_csv(path_to_file, index=False)

    def combine_results_summary_by_admitting_unit(self):
        """Group by admitting unit summary"""
        file_to_merge = self.setup.file_results_summary_by_admitting_unit

        data = self._stack_multiple_dataframes(file_to_merge)

        col_to_group = data.columns[0]
        cols_to_keep = ['utility_shift_mean', 'mRS shift_mean', 'mRS 0-2_mean']
        df = self._group_data_and_diff(data, col_to_group, cols_to_keep)

        # Save to file:
        output_dir = self.setup.dir_output_all_runs
        output_filename = self.setup.file_combined_results_summary_by_admitting_unit
        path_to_file = os.path.join(output_dir, output_filename)
        df.to_csv(path_to_file, index=False)


    # ############################
    # ##### HELPER FUNCTIONS #####
    # ############################
    def _stack_multiple_dataframes(self, file_to_merge):
        # Combine multiple DataFrames from different scenarios into here.
        # Stacks all DataFrames one on top of the other with no other
        # change in columns.
        data = pd.DataFrame()

        for d, dir_output in enumerate(self.setup.list_dir_output):
            file_input = file_to_merge
            path_to_file = os.path.join(dir_output, file_input)

            # Specify header to import as a multiindex DataFrame.
            df = pd.read_csv(path_to_file, header=[0, 1])

            first_column_name = df.columns[0][0]

            # Convert to normal index:
            df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
            # Rename the first column which didn't have multiindex levels:
            df = df.rename(columns={df.columns[0]: first_column_name})

            # Create a name for this scenario:
            scenario_name = os.path.split(dir_output)[-1]
            df['scenario'] = scenario_name
            data = pd.concat([data, df], axis=0)
        return data

    def _group_data_and_diff(self, data, col_to_group, cols_to_keep):
        """
        CHECK - is this function necessary? Are things already averaged before this?
        Do need to keep the diff part though.
        """
        # Combine data into this DataFrame:
        df = pd.DataFrame(columns=[col_to_group])
        scenario_name_list = sorted(list(set(data['scenario'])))
        # For scenarios [A, B, C], get a list of pairs [[A, B], [A, C], [B, C]].
        scenario_name_pairs = list(combinations(scenario_name_list, 2))

        mask_dict = dict(zip(
            scenario_name_list,
            [data['scenario'] == s for s in scenario_name_list]
        ))

        for c in cols_to_keep:
            cols = [col_to_group, c]
            # Copy over this column for each scenario
            # and where a column has multiple entries for one thing (e.g. LSOA),
            # then take the average value of the multiple entries.
            # (see Muster dev branch temp/)
            for scenario in scenario_name_list:
                data_new_col = data[
                    mask_dict[scenario]][cols].groupby(col_to_group).mean()
                data_new_col = data_new_col.reset_index()
                # Assume that this is a Series.
                data_new_col = data_new_col.rename(columns={c: f'{c}_{scenario}'})

                # Merge into the existing DataFrame and retain any empty rows.
                df = pd.merge(
                    df, data_new_col,
                    left_on=col_to_group, right_on=col_to_group,
                    how='outer'
                )

            # Take the difference between each pair of scenarios:
            for pair in scenario_name_pairs:
                p0 = pair[0]
                p1 = pair[1]
                diff_col_name = f'{c}_diff_{p0}_minus_{p1}'
                df[diff_col_name] = df[f'{c}_{p0}'] - df[f'{c}_{p1}']

        return df

    def _hstack_multiple_dataframes(self, file_to_merge):
        # Combine multiple DataFrames from different scenarios into here.
        # Stacks all DataFrames one on top of the other with no other
        # change in columns.
        data = pd.DataFrame()

        for d, dir_output in enumerate(self.setup.list_dir_output):
            file_input = file_to_merge
            path_to_file = os.path.join(dir_output, file_input)

            df = pd.read_csv(path_to_file, index_col=0)

            # Create a name for this scenario:
            scenario_name = os.path.split(dir_output)[-1]
            df = df.rename(columns=dict(zip(df.columns, [f'{c}_{scenario_name}' for c in df.columns])))
            data = pd.concat([data, df], axis=1)
        # Replace missing values with 0 in the region columns:
        data = data.fillna(value=0)
        data = data.convert_dtypes()
        return data

    def _merge_multiple_dataframes(self, file_to_merge, merge_col='LSOA11CD'):
        # Combine multiple DataFrames from different scenarios into here.
        # Stacks all DataFrames one on top of the other with no other
        # change in columns.
        data = pd.DataFrame(columns=[merge_col])
        scenario_cols_list = []
        scenario_series_list = []

        for d, dir_output in enumerate(self.setup.list_dir_output):
            file_input = file_to_merge
            path_to_file = os.path.join(dir_output, file_input)

            df = pd.read_csv(path_to_file)
            cols_order = df.columns.tolist()

            # Create a name for this scenario:
            scenario_name = os.path.split(dir_output)[-1]
            scenario_series = pd.Series(
                [1]*len(df),
                index=df[merge_col],
                name=scenario_name
            )

            scenario_cols_list.append(scenario_name)
            scenario_series_list.append(scenario_series)
            # Add all new LSOAs to the bottom of the existing DataFrame
            # and remove any duplicate rows.
            data = pd.concat([data, df], axis=0).drop_duplicates()
        # Merge in the Series data:
        for s in scenario_series_list:
            data = pd.merge(data, s.reset_index(), left_on=merge_col, right_on=merge_col, how='left')
        # Replace missing values with 0 in the scenario columns:
        data = data.fillna(value=dict(zip(scenario_cols_list, [0]*len(scenario_cols_list))))
        data = data.convert_dtypes()
        # Sort rows:
        data = data.sort_values(merge_col)
        return data
