"""
Part of the H9 project
"""
import numpy as np


def find_coords(target_rll, initial_mode, target_octants, h9_engine,
                projector_func, distance_func, depth=34, beam_width=6):
    """
    Finds the grid address for geographic points using a generic, vectorised beam search.
    This is a root-finding operation for projections that have an existing equilateral->ellipsoid projection
    but must depend upon root-finding for the inverse (ellipsoid->equilateral).

    Args:
        target_rll (np.ndarray): (N, 2) array of target [lat, lon] coordinates.
        initial_mode (np.ndarray): (N,) array of the starting mode for each point.
        target_octants (np.ndarray): (N, 3) array of the starting mode for each point.
        h9_engine: An h9_engine (or similar) instance providing the LUTs (ugc_lut, ugc_off) and constants.
        projector_func: A function that projects barycentric (x,y) to the target space (e.g., lat/lon).
        distance_func: A function that calculates the distance between two sets of target space coords.
        beam_width (int): The number of best candidates to keep at each layer.
        depth (int): The maximum depth of the address to generate.
        Beam Width:
            Accuracy vs. Speed: Increasing the beam width makes the search more robust and less likely to miss
            the correct path, especially for difficult edge cases.
            The trade-off is that a wider beam requires more computation at each layer, making the search slower.
            Number of Iterations: The beam width has no effect on the number of iterations.

    Returns:
        np.ndarray: (N, depth) array of the best URI address path found for each point.
    """
    # --- Initialisation ---
    # Start the search from the origin (0,0) with a beam width of 1.
    num_points = target_rll.shape[0]
    best_coords = np.zeros((num_points, 1, 2))
    root_uris = np.where(initial_mode == 1, 0x16, 0x49)
    best_paths = root_uris[:, np.newaxis, np.newaxis]

    # The loop variable `i` represents the current depth level (0, 1, 2...)
    for i in range(depth):
        current_beam_width = best_paths.shape[1]

        # --- a. Branch: Generate Candidates ---
        last_uris = best_paths[:, :, -1]
        parent_mode = h9_engine.ugc_lut[last_uris, h9_engine.mode]

        up_children = np.array(h9_engine.in_up_regions)
        down_children = np.array(h9_engine.in_dn_regions)
        next_gen_children_uris = np.where(parent_mode[..., np.newaxis] == 1, up_children, down_children)
        num_new_candidates = current_beam_width * 9
        next_gen_uris = next_gen_children_uris.reshape(num_points, num_new_candidates)  # Shape: (N, k * 9)

        # --- b. Evaluate: Incrementally Calculate Coordinates & Distances ---
        scale = (1 / 3) ** i
        parent_coords = np.repeat(best_coords, 9, axis=1)
        child_offsets = h9_engine.ugc_off[next_gen_uris]
        next_gen_coords_xy = parent_coords + child_offsets * scale

        # Project all candidates to lat/lon using the injected function
        num_candidates = next_gen_coords_xy.shape[1]
        tiled_components = np.repeat(target_octants[:, np.newaxis, :], num_candidates, axis=1)
        projected_rll = projector_func(next_gen_coords_xy, tiled_components)

        distances = distance_func(projected_rll, target_rll[:, np.newaxis, :])

        # --- c. Select: Prune to the Top `beam_width` Candidates ---
        best_indices = np.argsort(distances, axis=1)[:, :beam_width]

        # Update state for the next iteration
        best_uris = np.take_along_axis(next_gen_uris, best_indices, axis=1)
        best_coords = np.take_along_axis(next_gen_coords_xy, best_indices[:, :, np.newaxis], axis=1)

        # Reconstruct the winning paths
        parent_indices = best_indices // 9
        parent_paths = np.take_along_axis(best_paths, parent_indices[:, :, np.newaxis], axis=1)
        best_paths = np.concatenate([parent_paths, best_uris[:, :, np.newaxis]], axis=2)

    # The best address is the first candidate in the final set
    return best_coords[:, 0, :], best_paths[:, 0, :]
