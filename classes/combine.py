"""
Combine features from multiple runs.

Welcome to MultiIndex hell. *Doom trumpets*

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

        # Create the output folder for these combined files.
        self.dir_output_combined = self.setup.create_output_dir(
            self.setup.dir_output_combined, combined=True)

    def combine_files(self):
        self.combine_selected_regions()
        self.combine_selected_units()
        self.combine_selected_transfer()
        self.combine_selected_lsoa()
        self.combine_results_summary_by_lsoa()
        self.combine_results_summary_by_admitting_unit()

    # ##########################
    # ##### SPECIFIC FILES #####
    # ##########################

    def combine_selected_units(self, save_to_file=True):
        """
        Combine selected units.

        Each file input:
        +------+-----+---------+--------------+
        | Unit | ... | use_ivt |    coords    |   property
        +------+-----+---------+--------------+
        |    1 | ... |       1 | -x.xx, yy.yy |
        |    2 | ... |       1 | -x.xx, yy.yy |
        |    3 | ... |       1 | -x.xx, yy.yy |
        |  ... | ... |     ... |      ...     |
        |    n | ... |       0 | -x.xx, yy.yy |
        +------+-----+---------+--------------+

        Resulting DataFrame:
                                    +------------+------------+
                                    | scenario_1 | scenario_2 |    scenario
        +------+-----+--------------+------------+------------+
        | Unit | ... |    coords    |   use_ivt  |   use_ivt  |    property
        +------+-----+--------------+------------+------------+
        |    1 | ... | -x.xx, yy.yy |          1 |          0 |
        |    2 | ... | -x.xx, yy.yy |          1 |          0 |
        |    3 | ... | -x.xx, yy.yy |          1 |          1 |
        |  ... | ... |      ...     |        ... |        ... |
        |    n | ... | -x.xx, yy.yy |            |          1 |
        +------+-----+--------------+------------+------------+
        """
        file_to_merge = self.setup.file_selected_stroke_units

        try:
            df = self._hstack_multiple_dataframes(
                file_to_merge,
                cols_for_scenario=[
                    'use_ivt',
                    'use_mt',
                    'use_msu',
                    # 'Use',
                    'selected',
                    'transfer_unit_postcode',
                    'catches_lsoa_in_selected_region'
                ])
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property'])

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = self.setup.file_combined_selected_stroke_units
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    def combine_selected_transfer(self, save_to_file=True):
        """
        Combine selected units.

        Each file input:
        +------+-----+---------------+-----------------+
        | Unit | ... | transfer_unit | transfer_coords |    property
        +------+-----+---------------+-----------------+
        |    1 | ... |             2 |  -x.xx, yy.yy   |
        |    2 | ... |             2 |  -x.xx, yy.yy   |
        |    3 | ... |             2 |  -x.xx, yy.yy   |
        |  ... | ... |           ... |       ...       |
        |    n | ... |             4 |  -x.xx, yy.yy   |
        +------+-----+---------------+-----------------+

        Resulting DataFrame:
                               +------------+------------+
                               | scenario_1 | scenario_2 |    scenario
        +------+---------------+------------+------------+
        | Unit | transfer_unit |        Use |        Use |    property
        +------+---------------+------------+------------+
        |    1 |             1 |          1 |          0 |
        |    2 |             1 |          1 |          0 |
        |    3 |             1 |          1 |          1 |
        |  ... |           ... |        ... |        ... |
        |    1 |             9 |          0 |          1 |
        +------+---------------+------------+------------+
        """
        file_to_merge = self.setup.file_selected_transfer_units

        try:
            # Merge the separate files based on combo of unit and
            # transfer unit, two indexes.
            df = self._hstack_multiple_dataframes(
                file_to_merge,
                add_use_column=True,
                extra_cols_for_index=['name_nearest_mt']
                )
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property'])

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = self.setup.file_combined_selected_transfer_units
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    def combine_selected_lsoa(self, save_to_file=True):
        """
        Combine selected LSOA.

        Each file input:
        +------+-----+
        | LSOA | ... |    property
        +------+-----+
        |    1 | ... |
        |    2 | ... |
        |    3 | ... |
        |  ... | ... |
        |    n | ... |
        +------+-----+

        Resulting DataFrame:
               +--------------------+--------------------+
               |     scenario_1     |     scenario_2     |    scenario
        +------+-----+--------------+-----+--------------+
        | LSOA | Use | nearest_unit | Use | nearest_unit |    property
        +------+-----+--------------+-----+--------------+
        |    1 |   1 |            2 |   1 |            2 |
        |    2 |   1 |            2 |   0 |              |
        |    3 |   0 |            2 |   1 |            2 |
        |  ... | ... |          ... | ... |          ... |
        |    n |   0 |            9 |   0 |              |
        +------+-----+--------------+-----+--------------+
        """
        file_to_merge = self.setup.file_selected_lsoas

        try:
            df = self._hstack_multiple_dataframes( # TO DO - column name here might change
                file_to_merge, add_use_column=True, cols_for_scenario=['postcode_nearest_ivt'])
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property'])

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = self.setup.file_combined_selected_lsoas
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    def combine_selected_regions(self, save_to_file=True):
        """
        Combine selected regions.

        Each file input:
        +--------+-----------+----------+
        | Region | has_units | has_lsoa |    property
        +--------+-----------+----------+
        |      1 |      True |     True |
        |      2 |      True |     True |
        |      3 |     False |     True |
        |    ... |       ... |      ... |
        |      n |     False |     True |
        +--------+-----------+----------+

        Resulting DataFrame:
                 +----------------------+----------------------+
                 |      scenario_1      |      scenario_2      |    scenario
        +--------+-----------+----------+-----------+----------+
        | Region | has_units | has_lsoa | has_units | has_lsoa |    property
        +--------+-----------+----------+-----------+----------+
        |      1 |      True |     True |      True |     True |
        |      2 |      True |     True |     False |     True |
        |      3 |     False |     True |      True |     True |
        |    ... |       ... |      ... |       ... |      ... |
        |      n |     False |     True |     False |     True |
        +--------+-----------+----------+-----------+----------+
        """
        file_to_merge = self.setup.file_selected_regions

        try:
            df = self._hstack_multiple_dataframes(
                file_to_merge,
                csv_index=[0, 1],
                cols_for_scenario=[
                    'selected',
                    'contains_selected_lsoa',
                    'contains_unit_catching_lsoa'
                    ])
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property'])
        # Replace missing values with 0:
        # df = df.fillna(value=0)

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = self.setup.file_combined_selected_regions
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    def combine_results_summary_by_lsoa(self, save_to_file=True):
        """
        Group by LSOA summary.

        Each file input:
        +------+-------------+-------------+
        |      |   time_1    |    shift_1  |    property
        +------+------+------+------+------+
        | LSOA | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy |
        |    2 | x.xx | x.xx | y.yy | y.yy |
        |    3 | x.xx | x.xx | y.yy | y.yy |
        |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy |
        +------+------+------+------+------+

        Resulting DataFrame:
        +------+------+------+------+------+------+------+
        |      |  scenario_1 |  scenario_2 |    diff     |    scenario
        +------+------+------+------+------+------+------+
        |      |    shift    |    shift    |    shift    |    property
        +------+------+------+------+------+------+------+
        | LSOA | mean |  std | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    2 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    3 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        +------+------+------+------+------+------+------+
        """
        file_to_merge = self.setup.file_results_summary_by_lsoa

        try:
            data = self._hstack_multiple_dataframes(
                file_to_merge,
                csv_header=[0, 1],
                csv_index=[0, 1],
                cols_for_scenario=':'
                )
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # col_to_group = data.columns[0]
        cols_to_keep = ['utility_shift', 'mRS shift', 'mRS 0-2']
        # Same LSOA appearing in multiple files will currently have
        # multiple mostly-empty rows in the "data" DataFrame.
        # Group matching rows:
        # df = self._group_data(data, col_to_group, cols_to_keep)

        # Create new columns of this diff that:
        df = self._diff_data(data, cols_to_keep)

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property', 'subtype'])

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = self.setup.file_combined_results_summary_by_lsoa
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    def combine_results_summary_by_admitting_unit(self, save_to_file=True):
        """
        Group by admitting unit summary.

        Each file input:
        +------+-------------+-------------+
        |      |   time_1    |    shift_1  |    property
        +------+------+------+------+------+
        | Unit | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy |
        |    2 | x.xx | x.xx | y.yy | y.yy |
        |    3 | x.xx | x.xx | y.yy | y.yy |
        |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy |
        +------+------+------+------+------+

        Resulting DataFrame:
        +------+------+------+------+------+------+------+
        |  any |  scenario_1 |  scenario_2 |    diff     |    scenario
        +------+------+------+------+------+------+------+
        |      |    shift    |    shift    |    shift    |    property
        +------+------+------+------+------+------+------+
        | Unit | mean |  std | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    2 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    3 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        +------+------+------+------+------+------+------+
        """
        file_to_merge = self.setup.file_results_summary_by_admitting_unit

        try:
            data = self._hstack_multiple_dataframes(
                file_to_merge,
                csv_header=[0, 1],
                cols_for_scenario=':'
                )
        except FileNotFoundError:
            # TO DO - set up proper error message ----------------------------------
            pass

        # col_to_group = data.columns[0]
        cols_to_keep = ['utility_shift', 'mRS shift', 'mRS 0-2']
        # Same LSOA appearing in multiple files will currently have
        # multiple mostly-empty rows in the "data" DataFrame.
        # Group matching rows:
        # df = self._group_data(data, col_to_group, cols_to_keep)

        # Create new columns of this diff that:
        df = self._diff_data(data, cols_to_keep)

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property', 'subtype'])

        if save_to_file:
            output_dir = self.setup.dir_output_combined
            output_filename = (
                self.setup.file_combined_results_summary_by_admitting_unit)
            path_to_file = os.path.join(output_dir, output_filename)
            df.to_csv(path_to_file)#, index=False)

    # ############################
    # ##### HELPER FUNCTIONS #####
    # ############################
    def _diff_data(self, df, cols_to_diff):
        """
        C
        """
        # Combine data into this DataFrame:

        # Change to select top level of multiindex:
        scenario_name_list = sorted(list(set(
            df.columns.get_level_values(0).to_list())))
        try:
            # Drop 'any' scenario:
            scenario_name_list.remove('any')
        except ValueError:
            # no 'any' scenario here.
            pass

        # For scenarios [A, B, C], get a list of pairs
        # [[A, B], [A, C], [B, C]].
        scenario_name_pairs = list(combinations(scenario_name_list, 2))

        for c in cols_to_diff:
            # Take the difference between each pair of scenarios:
            for pair in scenario_name_pairs:
                p0 = pair[0]
                p1 = pair[1]
                diff_col_name = f'diff_{p0}_minus_{p1}'

                data0 = df[p0][c]
                data1 = df[p1][c]
                try:
                    for col in data0.columns:
                        if col in ['mean', 'median']:
                            # Take the difference between averages.
                            data_diff = data0[col] - data1[col]
                        elif col in ['std']:
                            # Propagate errors for std.
                            # Convert pandas NA to numpy NaN.
                            d0 = data0[col].copy().pow(2.0)
                            d1 = data1[col].copy().pow(2.0)
                            d2 = d0.add(d1, fill_value=0)
                            data_diff = d2.pow(0.5)
                            # data_diff = np.sqrt(np.nansum(
                            #     [data0[col]**2.0,  data1[col]**2.0]))
                        else:
                            # Don't know what to do with the rest yet. ----------------------
                            # TO DO
                            data_diff = ['help'] * len(df)
                        df[diff_col_name, c, col] = data_diff
                except AttributeError:
                    # No more nested column index levels.
                    data_diff = data0 - data1
                    df[diff_col_name, c] = data_diff
                    # TO DO - what about std herE? ---------------------------------
        return df

    def _hstack_multiple_dataframes(
            self,
            file_to_merge,
            csv_header=0,
            csv_index=0,
            add_use_column=False,
            cols_for_scenario=[],
            extra_cols_for_index=[]
            ):
        """
        # Combine multiple DataFrames from different scenarios into here.
        # Stacks all DataFrames one on top of the other with no other
        # change in columns.

        Each file input:
        +--------+-----------+----------+
        | Region | has_units | has_lsoa |
        +--------+-----------+----------+
        |      1 |      True |     True |
        |      2 |      True |     True |
        |      3 |     False |     True |
        |    ... |       ... |      ... |
        |      n |     False |     True |
        +--------+-----------+----------+

        Resulting DataFrame:
                 +----------------------+----------------------+
                 |      scenario_1      |      scenario_2      |
        +--------+-----------+----------+-----------+----------+
        | Region | has_units | has_lsoa | has_units | has_lsoa |
        +--------+-----------+----------+-----------+----------+
        |      1 |      True |     True |      True |     True |
        |      2 |      True |     True |     False |     True |
        |      3 |     False |     True |      True |     True |
        |    ... |       ... |      ... |       ... |      ... |
        |      n |     False |     True |     False |     True |
        +--------+-----------+----------+-----------+----------+
        """
        # data = pd.DataFrame()

        dfs_to_merge = {}

        for d, dir_output in enumerate(self.setup.list_dir_output):
            file_input = file_to_merge
            path_to_file = os.path.join(dir_output, file_input)

            df = pd.read_csv(
                path_to_file, index_col=csv_index, header=csv_header)

            if len(extra_cols_for_index) > 0:
                index_col = df.index.name
                df = df.reset_index()
                df = df.set_index([index_col] + extra_cols_for_index)

            if len(dfs_to_merge.items()) < 1:
                shared_col_name = df.index.name#columns[0]
                # dfs_to_merge['any'] = df[shared_col_name]

            # Create a name for this scenario:
            scenario_name = os.path.split(dir_output)[-1]

            if isinstance(cols_for_scenario, str):
                # Use all columns.
                pass
            else:
                split_for_any = True if ((len(cols_for_scenario) != (len(df.columns)))) else False
                if split_for_any:
                    # Find the names of these columns in this df.
                    # (so can specify one level of multiindex only).

                    scenario_cols = [self.find_multiindex_col(
                        df.columns, col) for col in cols_for_scenario]

                    if d == 0:
                        # Remove these columns from this scenario.
                        df_any = df.copy().drop(cols_for_scenario, axis='columns')
                        dfs_to_merge['any'] = df_any
                    else:
                        pass

                    # Remove columns for "any" scenario:
                    df = df[scenario_cols]

            if add_use_column:
                shared_col_is_list = ((type(shared_col_name) == list) | 
                                      (type(shared_col_name) == tuple))
                # TO DO ^ make this more generic.
                if shared_col_is_list:
                    use_col = tuple([b for b in shared_col_name[:-1]] + ['Use'])
                else:
                    use_col = 'Use'
                df[use_col] = 1

            dfs_to_merge[scenario_name] = df

        # Can't concat without index columns.
        data = pd.concat(
            dfs_to_merge.values(),
            axis='columns',
            keys=dfs_to_merge.keys()  # Names for extra index row
            )

        # data = data.reset_index()
        try:
            # Move 'any' index to the far left:
            cols = list(np.unique(data.columns.get_level_values(0).to_list()))
            cols.remove('any')
            cols = ['any'] + cols
            data = data[cols]
        except ValueError:
            # No 'any' columns yet.
            pass

        # Sort rows by contents of index:
        data = data.sort_index()

        # Did have dtype float/str from missing values, now want int:
        data = data.convert_dtypes()

        return data

    def _merge_multiple_dataframes(self, file_to_merge, merge_col='lsoa_code'):
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
            data = pd.merge(
                data, s.reset_index(),
                left_on=merge_col, right_on=merge_col, how='left'
                )
        # Replace missing values with 0 in the scenario columns:
        data = data.fillna(
            value=dict(zip(scenario_cols_list,
                           [0]*len(scenario_cols_list))))
        data = data.convert_dtypes()
        # Sort rows:
        data = data.sort_values(merge_col)
        return data


    def find_multiindex_col(self, columns, target):
        """
        MOVE ME - currently copied directly from Map()
        """
        if (type(columns[0]) == list) | (type(columns[0]) == tuple):
            # Convert all columns tuples into an ndarray:
            all_cols = np.array([[n for n in c] for c in columns])
        else:
            # No MultiIndex.
            all_cols = columns.values
        # Find where the grid matches the target string:
        inds = np.where(all_cols == target)
        # If more than one column, select the first.
        ind = inds[0][0]
        # Components of column containing the target:
        bits = all_cols[ind]
        bits_is_list = (type(columns[0]) == list) | (type(columns[0]) == tuple)
        # TO DO - make this generic arraylike ^
        # Convert to tuple for MultiIndex or str for single level.
        final_col = list((tuple(bits), )) if bits_is_list else bits
        return final_col