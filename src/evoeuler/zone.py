"""Zone-related functions."""

from __future__ import annotations

from itertools import combinations

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def all_zone_labels(set_labels: set[str]) -> set[str]:
    """Generate all possible zone labels from set labels.

    Args:
        set_labels: A set of strings representing the set labels.

    Returns:
        A set of strings representing all possible zone labels.
    """
    zone_labels = set()
    sorted_set_labels = sorted(set_labels)
    for combination_size in range(1, len(sorted_set_labels) + 1):
        for combination in combinations(sorted_set_labels, combination_size):
            zone_labels.add(" ".join(combination))
    return zone_labels


def extract_set_labels_from_zones(zones: set[str]) -> set[str]:
    """Extract a set of set labels from zone labels.

    Args:
        zones: A set of strings representing zone labels.

    Returns:
        A set of strings representing set labels.
    """
    sets = set()
    for zone_label in zones:
        sets.update(zone_label.strip().split())
    return sets


def compute_zone_geoms(
    shapes: dict[str, Polygon], zone_labels: set[str]
) -> dict[str, BaseGeometry]:
    """Compute the geometry of zones from shapes.

    Args:
        shapes: A dictionary mapping set labels to their Shapely Polygons.
        zone_labels: A set of strings representing the target zone labels.

    Returns:
        A dictionary mapping zone labels to their Shapely BaseGeometry.
    """
    zone_geoms: dict[str, BaseGeometry] = {}
    all_sets_labels = extract_set_labels_from_zones(zone_labels)
    # For each zone, compute its geometry.
    for zone_label in zone_labels:
        # Determine which sets should be included and excluded for this zone.
        included_set_labels = set(zone_label.split())
        excluded_set_labels = all_sets_labels - included_set_labels
        # Collect geometry of shapes for the included sets.
        included_shapes = []
        for included_set_label in sorted(included_set_labels):
            if included_set_label in shapes:
                included_shapes.append(shapes[included_set_label])
        # If no included shapes exist, create an empty GeometryCollection for
        # this zone.
        if not included_shapes:
            zone_geoms[" ".join(sorted(included_set_labels))] = (
                GeometryCollection()
            )
            continue
        # Initialize the zone geometry as the first included shape.
        zone_geom: BaseGeometry = included_shapes[0]
        # For each iteration, intersect the current zone geometry with an
        # included shape. If the resulting zone geometry is empty, break the
        # inner for loop early.
        for included_shape in included_shapes[1:]:
            zone_geom = zone_geom.intersection(included_shape)
            if zone_geom.is_empty:
                break
        # Skip the inner for loop if the current zone geometry is empty.
        if not zone_geom.is_empty:
            # For each iteration, subtract an excluded shape from the current
            # zone geometry. If the resulting zone geometry is empty, break
            # the inner for loop early.
            for excluded_set_label in sorted(excluded_set_labels):
                if excluded_set_label in shapes:
                    zone_geom = zone_geom.difference(shapes[excluded_set_label])
                    if zone_geom.is_empty:
                        break
        # Save the computed zone geometry.
        zone_geoms[" ".join(sorted(included_set_labels))] = zone_geom
    return zone_geoms


def get_connectivity_status(zone_geom: BaseGeometry) -> bool | None:
    """Get the connectivity status of a zone.

    Args:
        zone_geom: A Shapely BaseGeometry representing the zone.

    Returns:
        A boolean indicating whether the zone is connected or None if not
        applicable.
    """
    # Non-empty Polygons are always connected.
    if (
        isinstance(zone_geom, Polygon)
        and not zone_geom.is_empty
        and zone_geom.area > 0.0
    ):
        return True
    # Return None (N/A) if n_region == 0, True (Connected) if n_region == 1,
    # False (Disconnected) if n_region > 1.
    elif isinstance(zone_geom, MultiPolygon):
        n_region = 0
        for polygon in zone_geom.geoms:
            if not polygon.is_empty and polygon.area > 0.0:
                n_region += 1
        if n_region == 0:
            return None
        else:
            return n_region == 1
    elif isinstance(zone_geom, GeometryCollection):
        n_region = 0
        for geom in zone_geom.geoms:
            if (
                isinstance(geom, Polygon)
                and not geom.is_empty
                and geom.area > 0.0
            ):
                n_region += 1
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    if not polygon.is_empty and polygon.area > 0.0:
                        n_region += 1
        if n_region == 0:
            return None
        else:
            return n_region == 1
    # Return None (N/A) for other cases.
    else:
        return None


def format_connectivity_status(zone_geom: BaseGeometry) -> str:
    """Format the connectivity status of a zone.

    Args:
        zone_geom: A Shapely BaseGeometry representing the zone.

    Returns:
        A string representing the connectivity status ("Connected",
        "Disconnected (N parts)", or "N/A").
    """
    # Non-empty Polygons are always connected.
    if (
        isinstance(zone_geom, Polygon)
        and not zone_geom.is_empty
        and zone_geom.area > 0.0
    ):
        return "Connected"
    # Return "N/A" if n_region == 0, "Connected" if n_region == 1,
    # "Disconnected ({n_region} parts)" if n_region > 1.
    elif isinstance(zone_geom, MultiPolygon):
        n_region = 0
        for polygon in zone_geom.geoms:
            if not polygon.is_empty and polygon.area > 0.0:
                n_region += 1
        if n_region == 0:
            return "N/A"
        elif n_region == 1:
            return "Connected"
        else:
            return f"Disconnected ({n_region} parts)"
    elif isinstance(zone_geom, GeometryCollection):
        n_region = 0
        for geom in zone_geom.geoms:
            if (
                isinstance(geom, Polygon)
                and not geom.is_empty
                and geom.area > 0.0
            ):
                n_region += 1
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    if not polygon.is_empty and polygon.area > 0.0:
                        n_region += 1
        if n_region == 0:
            return "N/A"
        elif n_region == 1:
            return "Connected"
        else:
            return f"Disconnected ({n_region} parts)"
    # Return "N/A" for other cases.
    else:
        return "N/A"


def parse_zone_input(zone_input: str) -> set[str]:
    """Parse user zone input into a set of zone labels.

    Args:
        zone_input: A multi-line string with one zone label per line.

    Returns:
        A set of strings representing the parsed zone labels.
    """
    zones = set()
    for raw_line in zone_input.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        zone_label_tokens = line.split()
        sorted_zone_label = " ".join(sorted(zone_label_tokens))
        zones.add(sorted_zone_label)
    return zones
