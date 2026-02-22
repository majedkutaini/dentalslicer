import numpy as np
from scipy.ndimage import distance_transform_edt
from utils_geometry import find_point_fast, line_intersection_fast, intersect_margin_with_slice_advanced


def find_bottom_right(slice_data):
    return find_point_fast(slice_data, 'bottom_right')

def find_bottom_left(slice_data):
    return find_point_fast(slice_data, 'bottom_left')

def find_bottom(slice_data):
    return find_point_fast(slice_data, 'bottom')

def find_left(slice_data):
    return find_point_fast(slice_data, 'left')

def calculate_angle(a, b, c):
    """Calculate the angle between points a-b-c in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def calculate_parallel_line_distance(line1_points, line2_points, mm_per_pixel):
    """
    Calculate perpendicular distance between two parallel lines
    line1_points and line2_points are tuples of (start_point, end_point)
    """
    line1_start, line1_end = line1_points
    line2_start, line2_end = line2_points
    
    # Get direction vector of first line
    direction = np.array(line1_end) - np.array(line1_start)
    direction = direction / np.linalg.norm(direction)
    
    # Get perpendicular direction
    perp_direction = np.array([-direction[1], direction[0]])
    
    # Vector from line1 to line2
    to_line2 = np.array(line2_start) - np.array(line1_start)
    
    # Perpendicular distance
    distance_px = abs(np.dot(to_line2, perp_direction))
    distance_mm = distance_px * mm_per_pixel
    
    return distance_mm

def line_intersection(line1, line2):
    """Optimized line intersection wrapper"""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    x, y = line_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4)
    return (x, y) if x is not None else None

def first_hit_on_line(dist_map, line_start, line_end, max_dist_px, sample_step=0.5):
    """Return first point along line where distance map is within threshold."""
    start = np.array(line_start, dtype=float)
    end = np.array(line_end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return None
    direction /= length
    steps = int(length / sample_step)
    if steps <= 0:
        steps = 1
    height, width = dist_map.shape
    for i in range(steps + 1):
        p = start + direction * (i * sample_step)
        x = int(round(p[0]))
        y = int(round(p[1]))
        if 0 <= x < width and 0 <= y < height:
            if dist_map[y, x] <= max_dist_px:
                return (p[0], p[1])
    return None

def get_relevant_segments(contour_points_xy, base_left, base_right, perp_direction, max_steps, negative=True):
    """
    Filter contour segments to only those in the scanning area
    """
    if len(contour_points_xy) == 0:
        return contour_points_xy
    
    # Calculate bounding box of scanning area
    max_offset = max_steps
    direction = -perp_direction if negative else perp_direction
    
    # Four corners of scanning area
    corners = [
        base_left,
        base_right,
        base_left + direction * max_offset,
        base_right + direction * max_offset
    ]
    
    # Get bounding box with margin
    margin = 20
    min_x = min(corner[0] for corner in corners) - margin
    max_x = max(corner[0] for corner in corners) + margin
    min_y = min(corner[1] for corner in corners) - margin
    max_y = max(corner[1] for corner in corners) + margin
    
    # Filter points within bounding box
    relevant_points = []
    for point in contour_points_xy:
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            relevant_points.append(point)
    
    return np.array(relevant_points) if relevant_points else np.array([]) 

def calculate_mep_and_detection_points(slice_data, prep_slice_data, prep_k_slice_data, mm_per_pixel):
    """
    Unified method to calculate mep, sulcusboden, and col points.
    Returns: (final_mep, current_sulcusboden, current_vop, extended_line_points)
    """
   
    präpgrenze_left = find_bottom_left(prep_slice_data)
    präpgrenze_right = find_bottom_right(prep_slice_data)
    
    final_mep = None
    current_sulcusboden = None
    current_vop = None
    extended_line_points = None
    sulcus_distance_mm = None
    vop_distance_mm = None
    sulcustiefe = None
    shifted_left, shifted_right = None, None
    shifted_vop_left, shifted_vop_right = None, None
    
    # Calculate extended line and detection points
    if präpgrenze_left and präpgrenze_right:
        
        left = np.array(präpgrenze_left)
        right = np.array(präpgrenze_right)
        
        # Calculate direction vector
        direction = right - left
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Extended line for mep calculation
        extended_left = left - direction * 90
        extended_right = right + direction * 250
        extended_line_points = (extended_left, extended_right)
        
        # Length of VOP detection line!
        vop_left = right + direction * 30
        vop_right = right + direction * 250
        # Length of Sulcus floor detection line!
        new_left = left + direction * 600
        new_right = left + direction * 250
        
        # Get contour points
        contour_points = np.column_stack(np.where(slice_data > 0))
        if len(contour_points) > 0:
            contour_points_xy = contour_points[:, [1, 0]]  # Convert to (x,y)
            perp_direction = np.array([-direction[1], direction[0]])
            # Detection algorithm
            step_size = 1
            max_steps = 600
            dist_map = distance_transform_edt(~(slice_data > 0))
            vop_hit_epsilon = 0.75
   
            sulcus_segments = get_relevant_segments(contour_points_xy, new_left, new_right, perp_direction, max_steps, negative=True)
            vop_segments = get_relevant_segments(contour_points_xy, vop_left, vop_right, perp_direction, max_steps, negative=False)
            
            for step in range(0, max_steps):
                # === Sulcusfloor Detection ===
                reverse_step = max_steps - 1 - step
                sulcus_offset = -perp_direction * reverse_step * step_size
                shifted_left = new_left + sulcus_offset
                shifted_right = new_right + sulcus_offset
                
                for i in range(len(sulcus_segments)-1):
                    p1 = sulcus_segments[i]
                    p2 = sulcus_segments[i+1]
                    sulcus_intersect = line_intersection((shifted_left, shifted_right), (p1, p2))
                    if sulcus_intersect:
                        current_sulcusboden = sulcus_intersect
                        if extended_line_points:
                            sulcus_distance_mm = calculate_parallel_line_distance(
                                extended_line_points, 
                                (shifted_left, shifted_right),
                                mm_per_pixel
                            )
                        break
                
                if current_sulcusboden:
                    break

            for step in range(0, max_steps):
                # === VOP Detection ===
                reverse_step = max_steps - 1 - step
                vop_offset = perp_direction * reverse_step * step_size
                shifted_vop_left = vop_left + vop_offset
                shifted_vop_right = vop_right + vop_offset
                
                for i in range(len(vop_segments)-1):
                    p1 = vop_segments[i]
                    p2 = vop_segments[i+1]
                    if np.linalg.norm(p2 - p1) < 2:  # Only adjacent points
                        vop_intersect = line_intersection((shifted_vop_left, shifted_vop_right), (p1, p2))
                        if vop_intersect:
                            dist_to_p1 = np.linalg.norm(np.array(vop_intersect) - np.array(p1))
                            dist_to_p2 = np.linalg.norm(np.array(vop_intersect) - np.array(p2))
                            if dist_to_p1 < 5 and dist_to_p2 < 5:
                                current_vop = vop_intersect
                                # Calculate distance from extended line to vop detection line
                                if extended_line_points:
                                    vop_distance_mm = calculate_parallel_line_distance(
                                        extended_line_points, 
                                        (shifted_vop_left, shifted_vop_right),
                                        mm_per_pixel
                                    )
                                break
                if current_vop is None:
                    vop_intersect = first_hit_on_line(
                        dist_map,
                        shifted_vop_left,
                        shifted_vop_right,
                        vop_hit_epsilon,
                        sample_step=0.5
                    )
                    if vop_intersect:
                        current_vop = vop_intersect
                        if extended_line_points:
                            vop_distance_mm = calculate_parallel_line_distance(
                                extended_line_points,
                                (shifted_vop_left, shifted_vop_right),
                                mm_per_pixel
                            )
                
                if current_vop is not None:
                    break
    
    # === MEP (Margin Extrapolation Point) Calculation ===
    if extended_line_points is not None and slice_data is not None and slice_data.any():
        p1, p2 = extended_line_points          # keep as floats; do NOT round here
        mep = intersect_margin_with_slice_advanced(slice_data, p1, p2, ref_point=präpgrenze_right, min_dist_px=20, mode="first_after")
        if mep is not None:
            final_mep = mep
        else:
            final_mep = None

    
    if current_sulcusboden and current_vop is not None:
        sulcustiefe = calculate_parallel_line_distance(
                                        (shifted_left, shifted_right), 
                                        (shifted_vop_left, shifted_vop_right),
                                        mm_per_pixel
                                    )
    return final_mep, current_sulcusboden, current_vop, extended_line_points, (shifted_left, shifted_right), (shifted_vop_left, shifted_vop_right), sulcus_distance_mm, vop_distance_mm, sulcustiefe
