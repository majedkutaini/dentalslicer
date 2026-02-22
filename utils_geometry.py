import numpy as np
from numba import njit, prange

@njit(parallel=True)
def batch_line_intersections(line_starts, line_ends, contour_points):
    """
    Compute intersections between multiple scan lines and contour segments in parallel
    Returns intersection points and their corresponding line indices
    """
    intersections = []
    line_indices = []
    
    n_lines = len(line_starts)
    n_contour = len(contour_points) - 1
    
    for line_idx in prange(n_lines):
        x1, y1 = line_starts[line_idx]
        x2, y2 = line_ends[line_idx]
        
        for seg_idx in range(n_contour):
            x3, y3 = contour_points[seg_idx]
            x4, y4 = contour_points[seg_idx + 1]
            
            # Fast line intersection
            denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
            if abs(denom) > 1e-10:  # Not parallel
                ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
                ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
                
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    x = x1 + ua*(x2-x1)
                    y = y1 + ua*(y2-y1)
                    intersections.append((x, y))
                    line_indices.append(line_idx)
    
    return intersections, line_indices

@njit
def find_point_fast(slice_data, mode='bottom_right'):
    """Optimized point finding using numba JIT compilation"""
    points = np.where(slice_data > 0)
    if len(points[0]) == 0:
        return None
    
    rows, cols = points
    
    if mode == 'bottom_right':
        scores = cols - rows  # x - y for bottom-right
        best_idx = np.argmax(scores)
    elif mode == 'bottom_left':
        scores = -cols - rows  # -x - y for bottom-left
        best_idx = np.argmax(scores)
    elif mode == 'bottom':
        best_idx = np.argmin(rows)  # minimum Y
    elif mode == 'left':
        best_idx = np.argmax(rows)  # maximum Y
    else:
        best_idx = 0
    
    return (cols[best_idx], rows[best_idx])

@njit
def line_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4):
    """Fast line intersection using numba"""
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if abs(denom) < 1e-10:  # Lines are parallel
        return None, None
        
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua*(x2-x1)
        y = y1 + ua*(y2-y1)
        return x, y
    return None, None

@njit
def rasterize_line_fast(image, x1, y1, x2, y2):
    """Fast line rasterization using Bresenham's algorithm with numba"""
    height, width = image.shape
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    
    x, y = x1, y1
    
    if dx > dy:
        err = dx / 2
        while x != x2:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = 1.0
            err -= dy
            if err < 0:
                y += y_step
                err += dx
            x += x_step
    else:
        err = dy / 2
        while y != y2:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = 1.0
            err -= dx
            if err < 0:
                x += x_step
                err += dy
            y += y_step
    
    # Mark final point
    if 0 <= x2 < width and 0 <= y2 < height:
        image[y2, x2] = 1.0

def _bilinear(img, x, y):
    h, w = img.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, w-1), min(y0+1, h-1)
    dx, dy = x - x0, y - y0
    v00 = img[y0, x0]; v10 = img[y0, x1]; v01 = img[y1, x0]; v11 = img[y1, x1]
    return (v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy)

def _clamp_segment_to_image(p1, p2, h, w, eps=1e-6):
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = p2 - p1
    tmin, tmax = 0.0, 1.0
    for coord in range(2):
        p, dp = p1[coord], d[coord]
        lo, hi = 0.0, (w-eps if coord == 0 else h-eps)
        if abs(dp) < 1e-14:
            if p < lo or p > hi:
                return None
        else:
            t0, t1 = (lo - p)/dp, (hi - p)/dp
            t_enter, t_exit = min(t0, t1), max(t0, t1)
            tmin, tmax = max(tmin, t_enter), min(tmax, t_exit)
            if tmin > tmax:
                return None
    # pad a hair so we don't lose grazing intersections at the boundary
    pad = 1e-4
    return max(0.0, tmin - pad), min(1.0, tmax + pad)

def intersect_margin_with_slice(slice_data, p1, p2, prefer="last"):
    """
    Robust sub-pixel intersection of line segment (p1,p2) with the slice boundary.
    Works even when raster overlap is empty and across tiny angle changes.
    """
    field = (slice_data > 0).astype(np.float32)   # if you already have grayscale, use it instead

    h, w = field.shape
    rng = _clamp_segment_to_image(p1, p2, h, w)
    if rng is None:
        return None
    t0, t1 = rng

    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d  = p2 - p1
    L  = np.hypot(d[0], d[1])
    if L < 1e-9:
        return None

    def find_crossings(step):
        n = max(2, int(np.ceil((t1 - t0) * L / step)))
        ts = np.linspace(t0, t1, n)
        vals = np.array([_bilinear(field, *(p1 + t*d)) for t in ts]) - 0.5
        crossings = []
        for i in range(1, len(ts)):
            a, b = ts[i-1], ts[i]
            va, vb = vals[i-1], vals[i]

            # relaxed crossing test: any interval that straddles zero or touches it
            if (va == 0.0) or (vb == 0.0) or (va < 0) != (vb < 0):
                # refine by bisection
                lo, hi, vlo, vhi = a, b, va, vb
                for _ in range(30):
                    m = 0.5*(lo + hi)
                    vm = _bilinear(field, *(p1 + m*d)) - 0.5
                    # treat |vm| < tol as a hit to avoid endless flip-flop
                    if abs(vm) < 1e-8:
                        lo = hi = m
                        break
                    if (vlo < 0) != (vm < 0):
                        hi, vhi = m, vm
                    else:
                        lo, vlo = m, vm
                crossings.append(0.5*(lo + hi))
        return crossings

    # 1) coarse pass, then 2) fine pass if needed
    crossings = find_crossings(step=0.4)
    if not crossings:
        crossings = find_crossings(step=0.1)
    if not crossings:
        return None

    t = crossings[-1] if prefer == "last" else crossings[0]
    pt = p1 + t*d
    return float(pt[0]), float(pt[1])

def intersect_margin_with_slice_advanced(slice_data, p1, p2, ref_point=None, min_dist_px=0, mode="first_after"):
    """
    Find sub-pixel intersections of the segment (p1->p2) with the binary slice.
    """
    field = (slice_data > 0).astype(np.float32)

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    d = p2 - p1
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-6:
        return None
    u = d / seg_len  # unit direction

    # Sample the segment densely and detect threshold crossings (0 -> 1 or 1 -> 0)
    n_steps = max(2, int(seg_len * 2))  # 0.5px step
    ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    vals = []
    for t in ts:
        p = p1 + t * d
        x, y = p[0], p[1]
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if 0 <= ix < field.shape[1]-1 and 0 <= iy < field.shape[0]-1:
            # bilinear
            dx, dy = x - ix, y - iy
            v00 = field[iy, ix]
            v10 = field[iy, ix+1]
            v01 = field[iy+1, ix]
            v11 = field[iy+1, ix+1]
            v0 = v00*(1-dx) + v10*dx
            v1 = v01*(1-dx) + v11*dx
            v = v0*(1-dy) + v1*dy
        else:
            v = 0.0
        vals.append(v)
    vals = np.array(vals, dtype=np.float32)

    # detect zero-crossings around 0.5
    crossings = []
    for i in range(len(ts)-1):
        if (vals[i] - 0.5) * (vals[i+1] - 0.5) < 0:  # sign change across 0.5
            t0, t1 = ts[i], ts[i+1]
            v0, v1 = vals[i], vals[i+1]
            if abs(v1 - v0) < 1e-6:
                t = 0.5 * (t0 + t1)
            else:
                t = t0 + (0.5 - v0) * (t1 - t0) / (v1 - v0)
            pt = p1 + t * d
            crossings.append((t, pt))

    if not crossings:
        return None

    # If ref_point is given, compute its parametric t on the segment
    tref = None
    ref = None
    if ref_point is not None:
        ref = np.array(ref_point, dtype=np.float32)
        # project onto the segment
        w = ref - p1
        tref = float(np.dot(w, d) / (seg_len * seg_len))

    def valid_entry(entry):
        t, pt = entry
        if not (0.0 <= t <= 1.0):
            return False
        if ref is not None:
            # must be strictly AFTER the reference along the segment
            if t <= tref:
                return False
            # and far enough from the reference to ignore tiny immediate intersections
            if np.linalg.norm(pt - ref) < float(min_dist_px):
                return False
        return True

    candidates = [e for e in crossings if valid_entry(e)]
    if not candidates:
        return None

    if mode == "first_after":
        # pick the smallest t after tref
        candidates.sort(key=lambda e: e[0])
        t, pt = candidates[0]
        return float(pt[0]), float(pt[1])
    elif mode == "nearest_any":
        if ref is None:
            # fall back to first in param order if no reference
            candidates.sort(key=lambda e: e[0])
            t, pt = candidates[0]
            return float(pt[0]), float(pt[1])
        # pick by Euclidean distance to ref
        t, pt = min(candidates, key=lambda e: np.linalg.norm(e[1]-ref))
        return float(pt[0]), float(pt[1])
    else:
        # default to first_after behavior
        candidates.sort(key=lambda e: e[0])
        t, pt = candidates[0]
        return float(pt[0]), float(pt[1])
