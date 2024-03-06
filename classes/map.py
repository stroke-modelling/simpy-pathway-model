
    # #########################
    # ##### PLOT WRAPPERS #####
    # #########################
    def plot_map_selected_regions(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """
        # Check whether everything we need exists.
        data_for_prereqs = ['df_regions', 'df_units']
        self.load_run_data(data_for_prereqs)
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units']
        self.process_data(prereqs)
        self._check_prereqs_exist(prereqs)

        map_args, map_kwargs = self._setup_plot_map_selected_regions(
            scenario,
            save
            )
        self._plt_plot_map_selected_regions(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_selected_units(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """
        # Check whether everything we need exists.
        data_for_prereqs = ['df_regions', 'df_units', 'df_transfer']
        self.load_run_data(data_for_prereqs)
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_lines_transfer']
        self.process_data(prereqs)
        self._check_prereqs_exist(prereqs)

        map_args, map_kwargs = self._setup_plot_map_selected_units(
            scenario,
            save
            )
        self._plt_plot_map_selected_units(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_catchment(
            self,
            scenario: str,
            lsoa_catchment_type='',
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected unit catchments.
        """
        # Check whether everything we need exists.
        data_for_prereqs = ['df_regions', 'df_units',
                            'df_transfer', 'df_lsoa']
        self.load_run_data(data_for_prereqs)
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa', 'gdf_lines_transfer']
        self.process_data(prereqs)
        self._check_prereqs_exist(prereqs)

        map_args, map_kwargs = self._setup_plot_map_catchment(
            scenario,
            lsoa_catchment_type=lsoa_catchment_type,
            save=save
            )
        self._plt_plot_map_catchment(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            lsoa_catchment_type='',
            save=True,
            show=False,
            draw_region_boundaries=True,
            boundary_kwargs={},
            ):
        """
        Wrangle data and plot a map of LSOA outcomes.
        """
        # Check whether everything we need exists.
        data_for_prereqs = [
            'df_regions',
            'df_units',
            'df_transfer',
            'df_lsoa',
            'df_results_by_lsoa'
            ]
        self.load_run_data(data_for_prereqs)
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa']
        self.process_data(prereqs)
        self._check_prereqs_exist(prereqs)

        # TO DO - condense kwargs into dict. -----------------------------------------
        map_args, map_kwargs = self._setup_plot_map_outcome(
            scenario,
            outcome,
            lsoa_catchment_type=lsoa_catchment_type,
            boundary_kwargs=boundary_kwargs,
            save=save
            )
        self._plt_plot_map_outcome(
            *map_args,
            **map_kwargs,
            draw_region_boundaries=draw_region_boundaries,
            title=f'{scenario}\n{outcome}',
            save=save,
            show=show
        )

    # ###########################
    # ##### SETUP FOR PLOTS #####
    # ###########################
    def _setup_plot_map_selected_regions(
            self,
            scenario: str,
            save=True
            ):
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1])
        gdf_boundaries_regions = self._keep_only_geometry_in_box(
            gdf_boundaries_regions, box)
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)
        gdf_points_units = self._keep_only_geometry_in_box(
            gdf_points_units, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        gdf_boundaries_regions = self._assign_labels_and_points_to_regions(
            gdf_boundaries_regions,
            ['selected', 'BNG_N'], 'label', 'point_label')

        # Make a new column for whether stroke unit is in a selected
        # region.
        index_cols = gdf_boundaries_regions.index.names
        gdf_boundaries_regions = gdf_boundaries_regions.reset_index()
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        # gdf_boundaries_regions = gdf_boundaries_regions.set_index(index_cols)
        mask = gdf_points_units['region'].isin(regions_selected)
        gdf_points_units['region_selected'] = 0
        gdf_points_units.loc[mask, 'region_selected'] = 1

        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units, ['selected', 'region_selected', 'use_mt', 'BNG_N'])

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            'label',                                # label annotation
            'point_label',                          # label position
            'region',                               # legend label
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',         # point selection
            'stroke_team',                        # labels
            'label',                                # label annotation
            'selected'                              # ordering labels
            ]]

        # Create file name:
        if save:
            file_name = f'map_selected_regions_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_output_maps, file_name)
        else:
            path_to_file = None

        # TO DO  - save these args and kwargs and that to file ----------------------
        map_args = (
            gdf_boundaries_regions,
            gdf_points_units
            )
        map_kwargs = dict(
            map_extent=map_extent,
            path_to_file=path_to_file
            )
        return map_args, map_kwargs

    def _setup_plot_map_selected_units(
            self,
            scenario,
            save=True
            ):
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_lines_transfer']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
                gdf_lines_transfer = self.gdf_lines_transfer[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            gdf_lines_transfer = self._remove_excess_heading_from_gdf(
                gdf_lines_transfer, 0, 'geometry')
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()
            gdf_lines_transfer = self.gdf_lines_transfer.copy()

            # # The transfer lines data doesn't have a "Use" column here.
            gdf_lines_transfer['Use'] = 1
            # gdf_lines_transfer = self._find_use_column_for_transfer_lines(
            #     gdf_lines_transfer, gdf_points_units)

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1])
        gdf_boundaries_regions = self._keep_only_geometry_in_box(
            gdf_boundaries_regions, box)
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)
        gdf_points_units = self._keep_only_geometry_in_box(
            gdf_points_units, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units, ['selected', 'use_mt', 'BNG_N'])


        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',         # point selection
            'stroke_team',                          # labels
            'label',                                # label annotation
            'selected'                              # label kwargs
            ]]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
            ]]

        # Create file name:
        if save:
            file_name = f'map_selected_units_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_output_maps, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
        )
        map_kwargs = dict(
            path_to_file=path_to_file,
            map_extent=map_extent
            )
        return map_args, map_kwargs

    def _setup_plot_map_catchment(
            self,
            scenario: str,
            lsoa_catchment_type='',
            boundary_kwargs={},
            save=True
            ):
        """

        """
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa', 'gdf_lines_transfer']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_lsoa = self.gdf_boundaries_lsoa[c]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c]
                gdf_points_units = self.gdf_points_units[c]
                gdf_lines_transfer = self.gdf_lines_transfer[c]
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 0, ('geometry', ''))
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            gdf_lines_transfer = self._remove_excess_heading_from_gdf(
                gdf_lines_transfer, 0, 'geometry')
            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')
        else:
            gdf_boundaries_lsoa = self.gdf_boundaries_lsoa.copy()
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()
            gdf_lines_transfer = self.gdf_lines_transfer.copy()

            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')

            # The transfer lines data doesn't have a "Use" column here.
            gdf_lines_transfer['Use'] = 1

        # Combine LSOA geometry - from separate polygon per LSOA to one
        # big polygon for all LSOAs in catchment area.
        #  TO DO - might not always be called _ivt in following column:
        # TO DO - save this output file as .geojson for future use.
        gdf_boundaries_lsoa = self._combine_lsoa_into_catchment_shapes(
            gdf_boundaries_lsoa, 'unit_postcode')

        if lsoa_catchment_type == 'island':
            # Drop the 'contains_selected_lsoa' and
            # 'catches_lsoa_in_selected_region' columns.
            cols_to_keep_regions = []
            cols_to_keep_units = []

            # Only keep selected regions.
            mask = (gdf_boundaries_regions['selected'] == 1)
            gdf_boundaries_regions = gdf_boundaries_regions[mask]
            # Which stroke units are contained in this bounding box?
            mask = (gdf_points_units['selected'] == 1)
            gdf_points_units = gdf_points_units[mask]

            # Take map extent from the region geometry
            # *after* removing unwanted regions.
            box, map_extent = self.get_selected_area_extent(
                gdf_boundaries_regions)
        else:
            # Keep the 'contains_selected_lsoa' and
            # 'catches_lsoa_in_selected_region' columns.
            cols_to_keep_regions = [
                'contains_selected_lsoa',
                'contains_unit_catching_lsoa'
                ]
            cols_to_keep_units = ['catches_lsoa_in_selected_region']

            # Only keep stroke teams that contain LSOA in selected region.
            mask = gdf_points_units['catches_lsoa_in_selected_region'] == 1
            gdf_points_units = gdf_points_units[mask]

            # Take map extent from the combined LSOA, region,
            # and stroke unit geometry.
            region_mask = (
                (gdf_boundaries_regions['contains_selected_lsoa'] == 1) |
                (gdf_boundaries_regions['contains_unit_catching_lsoa'] == 1)
            )
            gdf_regions_reduced = gdf_boundaries_regions.copy()[region_mask
                ].reset_index()['geometry']
            gdf_lsoa_reduced = gdf_boundaries_lsoa.copy(
                ).reset_index()['geometry']
            gdf_units_reduced = gdf_points_units.copy(
                ).reset_index()['geometry']
            gdf_combo = pd.concat(
                (gdf_regions_reduced, gdf_lsoa_reduced, gdf_units_reduced),
                axis='rows')

            box, map_extent = self.get_selected_area_extent(gdf_combo)
            gdf_boundaries_regions = self._keep_only_geometry_in_box(
                gdf_boundaries_regions, box)

        # Restrict polygon geometry to the edges of the box.
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units, ['selected', 'use_mt', 'BNG_N'])


        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',                             # shapes
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            ] + cols_to_keep_regions                # line type selection
            ]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',  # 'Use',  # point selection
            'stroke_team',                          # labels
            'label',                                # label annotation
            'selected',                             # label kwargs
            ] + cols_to_keep_units                  # point selection
            ]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
            ]]

        lsoa_boundary_kwargs = {
            'cmap': 'Blues',
            'edgecolor': 'face'
        }
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            file_name = f'map_catchment_{scenario}'
            if len(lsoa_catchment_type) > 0:
                file_name += f'_{lsoa_catchment_type}'
            file_name += '.jpg'

            path_to_file = os.path.join(self.dir_output_maps, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
            )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            map_extent=map_extent,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    def _setup_plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            lsoa_catchment_type='',
            boundary_kwargs={},
            save=True
            ):
        """

        """
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa']
        self._check_prereqs_exist(prereqs)

        gdf_boundaries_lsoa = self.gdf_boundaries_lsoa.copy()
        gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
        gdf_points_units = self.gdf_points_units.copy()

        if self.data_type == 'combined':
            # Find shared outcome limits.
            # Take only scenarios containing 'diff':
            mask = gdf_boundaries_lsoa.columns.get_level_values(
                    0).str.contains('diff')
            if scenario.startswith('diff'):
                pass
            else:
                # Take the opposite condition, take only scenarios
                # not containing 'diff'.
                mask = ~mask

            mask = (
                mask &
                (gdf_boundaries_lsoa.columns.get_level_values(2) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(1) == outcome)
            )
            all_mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = all_mean_vals.abs().max().values[0]
            vmax = all_mean_vals.max().values[0]
            vmin = all_mean_vals.min().values[0]

            if 'diff' in scenario:
                # Add any other columns that these expect.
                gdf_boundaries_regions = self.create_combo_cols(
                    gdf_boundaries_regions, scenario)
                gdf_points_units = self.create_combo_cols(
                    gdf_points_units, scenario)

            try:
                # Remove excess scenario data:
                s = ['any', scenario]
                gdf_boundaries_lsoa = gdf_boundaries_lsoa[s]
                gdf_boundaries_regions = gdf_boundaries_regions[s]
                gdf_points_units = gdf_points_units[s]
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the "std" columns.
            mask = (gdf_boundaries_lsoa.columns.get_level_values(2) == 'std')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, mask].columns,
                axis='columns'
            )

            # Remove the 'scenario' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 0, ('geometry', ''))
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')
        else:
            # Find the colour limits.
            mask = (
                (gdf_boundaries_lsoa.columns.get_level_values(1) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(0) == outcome)
            )
            mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = mean_vals.abs().max().values[0]
            vmax = mean_vals.max().values[0]
            vmin = mean_vals.min().values[0]

            # Remove the "std" columns.
            mask = (gdf_boundaries_lsoa.columns.get_level_values(1) == 'std')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, mask].columns,
                axis='columns'
            )

            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')

        if lsoa_catchment_type == 'island':
            # Drop the 'contains_selected_lsoa' and
            # 'catches_lsoa_in_selected_region' columns.
            cols_to_keep_regions = []

            # Only keep selected regions.
            mask = (gdf_boundaries_regions['selected'] == 1)
            gdf_boundaries_regions = gdf_boundaries_regions[mask]

            # Take map extent from the region geometry
            # *after* removing unwanted regions.
            box, map_extent = self.get_selected_area_extent(
                gdf_boundaries_regions,
                leeway=20000,
                )
        else:
            # Keep the 'contains_selected_lsoa' and
            # 'catches_lsoa_in_selected_region' columns.
            cols_to_keep_regions = ['contains_selected_lsoa']

            # Take map extent from the combined LSOA and region geometry.
            mask_region = (
                (gdf_boundaries_regions['selected'] == 1) |
                (gdf_boundaries_regions['contains_selected_lsoa'] == 1)
                )
            gdf_regions_reduced = gdf_boundaries_regions.copy()[
                mask_region].reset_index()['geometry']
            gdf_lsoa_reduced = gdf_boundaries_lsoa.copy(
                ).reset_index()['geometry']
            gdf_combo = pd.concat(
                (gdf_regions_reduced, gdf_lsoa_reduced), axis='rows')

            box, map_extent = self.get_selected_area_extent(
                gdf_combo,
                leeway=20000,
                )
            gdf_boundaries_regions = self._keep_only_geometry_in_box(
                gdf_boundaries_regions, box)

        # Only keep selected stroke units.
        mask = (gdf_points_units['selected'] == 1)
        gdf_points_units = gdf_points_units[mask]

        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units, ['selected', 'use_mt', 'BNG_N'])

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',                             # shapes
            outcome                                 # colours
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            ] + cols_to_keep_regions                # line type selection
            ]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',# 'Use',  # point selection
            'stroke_team',                          # labels
            'label',                                # label annotation
            'selected'                              # label kwargs
            ]]

        lsoa_boundary_kwargs = {
            'column': outcome,
            'edgecolor': 'face',
            # Adjust size of colourmap key, and add label
            'legend_kwds': {
                'shrink': 0.5,
                'label': outcome
                },
            # Set to display legend
            'legend': True,
            }

        cbar_diff = True if scenario.startswith('diff') else False
        if cbar_diff:
            lsoa_boundary_kwargs['cmap'] = 'seismic'
            lsoa_boundary_kwargs['vmin'] = -vlim_abs
            lsoa_boundary_kwargs['vmax'] = vlim_abs
        else:
            cmap = 'plasma'
            if outcome == 'mRS shift':
                # Reverse the colourmap because lower values
                # are better for this outcome.
                cmap += '_r'
            lsoa_boundary_kwargs['cmap'] = cmap
            lsoa_boundary_kwargs['vmin'] = vmin
            lsoa_boundary_kwargs['vmax'] = vmax
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            file_name = f'map_outcome_{outcome}_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_output_maps, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units
        )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    # #######################
    # ##### PYPLOT MAPS #####
    # #######################
    def _plt_plot_map_selected_regions(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax, extra_artists = maps.plot_map_selected_regions(
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            map_extent=map_extent
        )

        if save:
            # Return extra artists so that bbox_inches='tight' line
            # in savefig() doesn't cut off the legends.
            # Adding legends with ax.add_artist() means that the
            # bbox_inches='tight' line ignores them.
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[1], side='left')
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='right')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_selected_units(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax, extra_artists = maps.plot_map_selected_units(
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_catchment(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            title='',
            lsoa_boundary_kwargs={},
            map_extent=[],
            save=True,
            show=False,
            path_to_file=''
            ):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(title)

        ax, extra_artists = maps.plot_map_catchment(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
            boundary_kwargs=lsoa_boundary_kwargs,
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight')
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_outcome(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            title='',
            lsoa_boundary_kwargs={},
            draw_region_boundaries=True,
            save=True,
            show=False,
            path_to_file=None
            ):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title)

        ax, extra_artists = maps.plot_map_outcome(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            boundary_kwargs=lsoa_boundary_kwargs,
            draw_region_boundaries=draw_region_boundaries,
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight')
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()
