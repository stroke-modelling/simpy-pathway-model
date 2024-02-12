"""
Functions for drawing maps.
"""
import matplotlib.pyplot as plt


# ####################
# ##### PLOTTING #####
# ####################
# n.b. The following functions mostly just use plt.plot()
# but are given different wrappers anyway for the sake of
# applying some kwargs automatically.
def draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt={},
        kwargs_with_lsoa={},
        kwargs_with_unit={}
        ):
    # Set up kwargs.
    kwargs_nowt = {
        'edgecolor': 'none',
        'facecolor': 'none',
    }
    kwargs_lsoa = {
        'edgecolor': 'silver',
        'facecolor': 'none',
        'linewidth': 0.5,
        'linestyle': '--'
    }
    kwargs_unit = {
        'edgecolor': 'k',
        'facecolor': 'none',
        'linewidth': 0.5
    }
    # Update these with anything from the input dicts:
    kwargs_nowt = kwargs_nowt | kwargs_with_nowt
    kwargs_lsoa = kwargs_lsoa | kwargs_with_lsoa
    kwargs_unit = kwargs_unit | kwargs_with_unit

    # Regions containing neither LSOAs nor stroke units here:
    mask = (
        (gdf_boundaries_regions['contains_selected_unit'] == 0) &
        (gdf_boundaries_regions['contains_selected_lsoa'] == 0)
    )
    gdf_boundaries_with_nowt = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_nowt) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_nowt,
            **kwargs_nowt
            )
        # Make these invisible but draw them anyway to make sure the
        # extent of the map is similar to other runs.
    else:
        pass

    # Regions containing LSOAs but not stroke units:
    mask = (
        (gdf_boundaries_regions['contains_selected_unit'] == 0) &
        (gdf_boundaries_regions['contains_selected_lsoa'] == 1)
    )
    gdf_boundaries_with_lsoa = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_lsoa) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_lsoa,
            **kwargs_lsoa
            )
    else:
        pass

    # Regions containing stroke units:
    mask = (gdf_boundaries_regions['contains_selected_unit'] == 1)
    gdf_boundaries_with_units = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_units) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_units,
            **kwargs_unit
            )
    else:
        pass
    return ax


def draw_boundaries(ax, gdf, **kwargs):
    """
    Draw regions from a GeoDataFrame.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the regions.
    gdf    - GeoDataFrame. Stores geometry to be plotted.
    kwargs - dict. Keyword arguments to pass to plt.plot().

    Returns
    -------
    ax - pyplot axis. Same as input but with regions drawn on.
    """
    # Draw the main map with colours (choropleth):
    gdf.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        **kwargs
        )
    return ax


def scatter_ivt_units(ax, gdf):
    """
    Draw scatter markers for IVT stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker for each hospital:
    mask = gdf['Use_IVT'] == 1
    IVT = gdf[mask]
    IVT.plot(
        ax=ax,
        edgecolor='k',
        facecolor='w',
        markersize=50,
        marker='o',
        zorder=2
        )
    return ax


def scatter_mt_units(ax, gdf):
    """
    Draw scatter markers for MT stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker star for MT units:
    mask = gdf['Use_MT'] == 1
    MT = gdf[mask]
    MT.plot(
        ax=ax,
        edgecolor='k',
        facecolor='y',
        markersize=300,
        marker='*',
        zorder=2
        )
    return ax


def scatter_msu_units(ax, gdf):
    """
    Draw scatter markers for MSU stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker star for MT/MSU units:
    mask = gdf['Use_MSU'] == 1
    MSU = gdf[mask]
    MSU.plot(
        ax=ax,
        edgecolor='k',
        facecolor='orange',
        markersize=50,
        marker='s',
        zorder=2
        )
    return ax


def plot_lines_between_units(ax, gdf):
    """
    Draw lines from stroke units to their MT transfer units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores LineString objects that connect
            each stroke unit to its MT transfer unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Draw a line connecting each unit to its MT unit.
    mask = gdf['Use'] == 1
    lines = gdf[mask]
    lines.plot(
        ax=ax,
        edgecolor='k',
        linestyle='-',
        linewidth=3,
        zorder=1  # Place it beneath the scatter markers.
    )
    return ax


def annotate_unit_labels(ax, gdf):
    """
    Draw label for each stroke unit.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores coordinates and name of each
            stroke unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    try:
        mask = gdf['labels_mask']
        gdf_labels = gdf[mask]
    except KeyError:
        # No mask column was given.
        gdf_labels = gdf

    # Define "z" to shorten following "for" line:
    z = zip(
        gdf_labels.geometry.x,
        gdf_labels.geometry.y,
        gdf_labels.Hospital_name
        )
    for x, y, label in z:
        # Edit the label to put a space in the postcode when displayed:
        label = f'{label[:-3]} {label[-3:]}'
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        ax.annotate(
            label, xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(facecolor='w', edgecolor='k'),
            fontsize=8
            )
    return ax


# ######################
# ##### MAIN PLOTS #####
# ######################
def plot_map_selected_units(
        gdf_boundaries_regions,  # TO DO - make this optional?
        gdf_points_units,
        gdf_lines_transfer,
        ax=None
    ):
    """
    Make map of the selected units and the regions containing them.

    Properties of this map:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + Postcode
        - for unit name matching.
        - for labels on the map.
    + Use_MT
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.

    Result is saved as the name given in setup.file_selected_units_map.
    """

    # Set up kwargs
    kwargs_with_nowt = {}
    kwargs_with_lsoa = {}
    kwargs_with_unit = {'facecolor': gdf_boundaries_regions['colour']}

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax = draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt=kwargs_with_nowt,
        kwargs_with_lsoa=kwargs_with_lsoa,
        kwargs_with_unit=kwargs_with_unit
        )
    ax = scatter_ivt_units(ax, gdf_points_units)
    ax = scatter_mt_units(ax, gdf_points_units)
    # ax = scatter_msu_units(ax, gdf_points_units)  # Not for Optimist
    ax = plot_lines_between_units(ax, gdf_lines_transfer)

    # Keep track of which units to label in here:
    gdf_points_units['labels_mask'] = False
    gdf_points_units.loc[
        gdf_points_units['Use'] == 1, 'labels_mask'] = True
    ax = annotate_unit_labels(ax, gdf_points_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    return ax


def plot_map_catchment(
        gdf_boundaries_lsoa,
        gdf_boundaries_regions,  # TO DO - make this optional
        gdf_points_units,
        gdf_lines_transfer,
        ax=None,
        boundary_kwargs={}
        ):
    """
    Map the selected units, containing regions, and catchment areas.

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + Postcode
        - for unit name matching.
        - for labels on the map.
    + Use_MT
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.
    + geojson file of LSOA boundaries.
    Must contain:
    + coordinates of each LSOA boundary.
    + selected LSOA name file.
    Must contain:
    + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
    Must contain:
    + column LSOA11CD for name matching.
    + postcode_nearest_IVT
    + postcode_nearest_MT
    + postcode_nearest_MSU

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
    """
    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))

    # LSOAs:
    ax = draw_boundaries(
        ax, gdf_boundaries_lsoa,
        **boundary_kwargs
        )

    # Background regions:
    ax = draw_boundaries_by_contents(ax, gdf_boundaries_regions)

    # Stroke unit markers.
    ax = scatter_ivt_units(ax, gdf_points_units)
    ax = scatter_mt_units(ax, gdf_points_units)
    # ax = scatter_msu_units(ax, gdf_points_units)  # Not for Optimist
    # Keep track of which units to label in here:
    gdf_points_units['labels_mask'] = False
    gdf_points_units.loc[
        gdf_points_units['Use'] == 1, 'labels_mask'] = True

    # Transfer unit lines.
    ax = plot_lines_between_units(ax, gdf_lines_transfer)

    # Stroke unit labels.
    ax = annotate_unit_labels(ax, gdf_points_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    return ax


def plot_map_outcome(
        gdf_boundaries_lsoa,
        gdf_boundaries_regions,  # TO DO - make this optional
        gdf_points_units,
        ax=None,
        boundary_kwargs={}
        ):
    """
    Map the selected units, containing regions, and catchment areas.

    UPDATE ME

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + Postcode
        - for unit name matching.
        - for labels on the map.
    + Use_MT
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.
    + geojson file of LSOA boundaries.
    Must contain:
    + coordinates of each LSOA boundary.
    + selected LSOA name file.
    Must contain:
    + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
    Must contain:
    + column LSOA11CD for name matching.
    + postcode_nearest_IVT
    + postcode_nearest_MT
    + postcode_nearest_MSU

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
    """
    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))

    # LSOAs:
    ax = draw_boundaries(
        ax, gdf_boundaries_lsoa,
        **boundary_kwargs
        )

    # Background regions:
    ax = draw_boundaries_by_contents(ax, gdf_boundaries_regions)

    # Stroke unit markers.
    ax = scatter_ivt_units(ax, gdf_points_units)
    ax = scatter_mt_units(ax, gdf_points_units)
    # ax = scatter_msu_units(ax, gdf_points_units)  # Not for Optimist
    # Keep track of which units to label in here:
    gdf_points_units['labels_mask'] = False
    gdf_points_units.loc[
        gdf_points_units['Use'] == 1, 'labels_mask'] = True

    # # Stroke unit labels.
    # ax = annotate_unit_labels(ax, gdf_points_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    return ax
