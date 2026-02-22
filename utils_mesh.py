import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from vedo import Plotter, Mesh, Point
from utils_geometry import rasterize_line_fast

class OptimizedMeshProcessor:
    def __init__(self, shared_bounds, slice_resolution):
        self.shared_bounds = shared_bounds
        self.slice_resolution = slice_resolution
        self.intersection_cache = {}  # Cache mesh intersections
        self.bounds_cache = {}        # Cache transformed bounds
        
        # Pre-calculate transformation matrices for common angles
        self.precomputed_transforms = {}
        
        # Calculate bounds once
        self.bounds_2d = np.array([
            [shared_bounds[0][1], shared_bounds[0][2]],  # Y_min, Z_min
            [shared_bounds[1][1], shared_bounds[1][2]]   # Y_max, Z_max
        ])
        
        # Pre-calculate scaling factors (keep square pixels for 1:1 mm)
        y_range = max(self.bounds_2d[1][0] - self.bounds_2d[0][0], 1e-6)
        z_range = max(self.bounds_2d[1][1] - self.bounds_2d[0][1], 1e-6)
        max_range = max(y_range, z_range)
        center_yz = (self.bounds_2d[0] + self.bounds_2d[1]) / 2.0
        half = max_range / 2.0
        self.bounds_2d = np.array([
            [center_yz[0] - half, center_yz[1] - half],
            [center_yz[0] + half, center_yz[1] + half],
        ])

        self.y_range = max_range
        self.z_range = max_range
        self.y_scale = (slice_resolution - 1) / max_range
        self.z_scale = (slice_resolution - 1) / max_range
    
    def get_plane_intersection_cached(self, mesh, angle, plane_origin, plane_normal):
        cache_key = (id(mesh), angle, tuple(plane_origin), tuple(plane_normal))
        if cache_key not in self.intersection_cache:
            result = self.get_plane_intersection(mesh, plane_origin, plane_normal)
            self.intersection_cache[cache_key] = result
        return self.intersection_cache[cache_key]
    
    def get_rotation_matrix(self, angle_degrees):
        """Cache rotation matrices for better performance"""
        if angle_degrees not in self.precomputed_transforms:
            angle_rad = np.deg2rad(angle_degrees)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            self.precomputed_transforms[angle_degrees] = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        return self.precomputed_transforms[angle_degrees]
    
    def rasterize_lines_optimized(self, lines_list):
        """Optimized line rasterization"""
        image = np.zeros((self.slice_resolution, self.slice_resolution), dtype=np.uint8)
        
        if not lines_list:
            return np.zeros((self.slice_resolution, self.slice_resolution), dtype=np.float32)
        
        for lines in lines_list:
            if len(lines) < 2:
                continue
            
            # Convert to image coordinates
            y_coords = ((lines[:, 0] - self.bounds_2d[0][0]) * self.y_scale).astype(np.int32)
            z_coords = ((lines[:, 1] - self.bounds_2d[0][1]) * self.z_scale).astype(np.int32)
            
            # Clamp coordinates
            y_coords = np.clip(y_coords, 0, self.slice_resolution - 1)
            z_coords = np.clip(z_coords, 0, self.slice_resolution - 1)
            
            # Draw lines using optimized function
            for i in range(len(y_coords) - 1):
                rasterize_line_fast(image, y_coords[i], z_coords[i], 
                                  y_coords[i+1], z_coords[i+1])
        
        return image

def best_fit_transform(A, B):
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()
    return T

def nearest_neighbor(src, dst):
    assert src.shape == dst.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def iterative_closest_point(A, B, max_iterations=20, tolerance=0.001):
    assert A.shape == B.shape
    m = A.shape[1]
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    prev_error = 0
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T = best_fit_transform(A, src[:m,:].T)
    final_error = prev_error
    rot = T[0:-1,0:-1]
    t = T[:-1,-1]
    finalA = np.dot(rot, A.T).T + t
    return T, finalA, final_error, i

def align_meshes_interactively(prep_mesh, prep_k_mesh, tooth_id, v, xx):
    """Align prep_k_mesh to prep_mesh using ICP with interactive point selection."""
    # Convert trimesh objects to vedo meshes
    prep_mesh_vedo = Mesh([prep_mesh.vertices, prep_mesh.faces]).color("red").legend("Target Mesh")
    prep_k_mesh_vedo = Mesh([prep_k_mesh.vertices, prep_k_mesh.faces]).color("blue").legend("Source Mesh")

    # Create a vedo Plotter for interactive point selection
    plt = Plotter(title="Left: Red (Dig), Right: Blue (Konv)")

    # Lists to store selected points
    source_points = []
    target_points = []
    drawn_points = []

    # Callback function to select points on the source mesh
    def select_source_point(evt):
        if evt.actor and evt.actor == prep_k_mesh_vedo:  # Ensure the click is on the blue model
            point = evt.picked3d
            source_points.append(point)
            print(f"Selected source point: {point}")
            
            # Draw a blue dot at the selected point
            dot = Point(point, c="red", r=10)
            drawn_points.append(dot)
            plt.add(dot)

    # Callback function to select points on the target mesh
    def select_target_point(evt):
        if evt.actor and evt.actor == prep_mesh_vedo:  # Ensure the click is on the red model
            point = evt.picked3d
            target_points.append(point)
            print(f"Selected target point: {point}")
            
            # Draw a red dot at the selected point
            dot = Point(point, c="blue", r=10)
            drawn_points.append(dot)
            plt.add(dot)

    def delete_last_point(evt):
        if evt.keypress == "Backspace":  # Only react to the Backspace key
            if drawn_points:
                last_point = drawn_points.pop()
                plt.remove(last_point)
                
                # Remove the last point from the corresponding list
                if source_points and np.array_equal(source_points[-1], last_point.pos()):
                    source_points.pop()
                elif target_points and np.array_equal(target_points[-1], last_point.pos()):
                    target_points.pop()
                print("Last point deleted.")
                plt.render() 

    # Add meshes to the plotter
    plt.add(prep_mesh_vedo)
    plt.add(prep_k_mesh_vedo)

    # Set up callbacks for point selection
    plt.add_callback('LeftButtonPress', select_source_point)
    plt.add_callback('RightButtonPress', select_target_point)
    plt.add_callback('KeyPress', delete_last_point)  # Bind delete key to delete_last_point

    # Show the interactive viewer
    plt.show()

    # Convert selected points to numpy arrays
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # Ensure at least 3 points are selected for alignment
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("Please select at least 3 corresponding points on both meshes.")

    # Save the selected points to a file
    icp_points = {
        "source_points": source_points,
        "target_points": target_points,
        "is_flipped": False
    }
    icp_filename = f"Zähne/{v}/{xx}/icp_points_{tooth_id}.npy"
    np.save(icp_filename, icp_points)
    print(f"ICP points saved to {icp_filename}")

    # Compute initial transformation using manually selected points
    initial_T = best_fit_transform(source_points, target_points)

    # Apply the initial transformation to the source mesh
    prep_k_mesh.apply_transform(initial_T)

    # Sample points from both meshes for ICP
    target_points_icp = prep_mesh.sample(5000)  # Points from präpgrenze (target)
    source_points_icp = prep_k_mesh.sample(5000)  # Points from präpgrenze_k (source)

    # Run custom ICP to align präpgrenze_k to präpgrenze
    T, finalA, final_error, iterations = iterative_closest_point(
        source_points_icp, 
        target_points_icp,
        max_iterations=500,  # Increase iterations for better alignment
        tolerance=1e-6       # Smaller tolerance for finer alignment
    )

    # Apply the ICP transformation to the entire präpgrenze_k mesh
    prep_k_mesh.apply_transform(T)

    # Assign darker colors to the meshes
    prep_mesh_color = [128, 0, 0, 255]  # Dark red for präpgrenze (RGBA)
    prep_k_mesh_color = [0, 0, 128, 255]  # Dark blue for präpgrenze_k (RGBA)

    # Apply colors to the meshes
    prep_mesh.visual.face_colors = prep_mesh_color
    prep_k_mesh.visual.face_colors = prep_k_mesh_color

    # Convert trimesh objects to vedo meshes
    prep_mesh_vedo = Mesh([prep_mesh.vertices, prep_mesh.faces]).color("red").legend("Target Mesh")
    prep_k_mesh_vedo = Mesh([prep_k_mesh.vertices, prep_k_mesh.faces]).color("blue").legend("Source Mesh")

    # Create a vedo Plotter
    plt = Plotter(title="Aligned Meshes")
    plt.add(prep_mesh_vedo)
    plt.add(prep_k_mesh_vedo)
    plt.show()  # This will block until the window is closed

    # Print alignment metrics
    print(f"Final alignment error: {final_error:.6f}")
    print(f"Number of iterations: {iterations}")
    return prep_k_mesh

def adjust_mesh_to_bounds(mesh, bounds):
    """Translate and scale the mesh to fit within the specified bounds."""
    # Calculate the current bounds of the mesh
    mesh_bounds = mesh.bounds
    
    # Calculate the required translation and scaling
    translation = (bounds[0] - mesh_bounds[0])  # Translate to align min bounds
    scaling = (bounds[1] - bounds[0]) / (mesh_bounds[1] - mesh_bounds[0])  # Scale to fit within bounds
    
    # Apply translation and scaling to the mesh
    mesh.apply_translation(translation)
    mesh.apply_scale(scaling)
    
    return mesh
