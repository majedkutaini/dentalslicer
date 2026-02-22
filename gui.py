import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
from multiprocessing import cpu_count
import tensorflow as tf
from tensorflow.keras import layers, models
from vedo import Plotter, Mesh, Plane
import csv

from utils_mesh import OptimizedMeshProcessor, align_meshes_interactively, best_fit_transform, iterative_closest_point
from utils_geometry import find_point_fast
from tooth_metrics import (
    calculate_mep_and_detection_points,
    find_bottom_left, find_bottom_right,
    line_intersection
)

class InteractiveRotationViewer:
    def __init__(self, model_file, prep_file, prep_k_file, v="M1", xx="11", num_angles=360, slice_resolution=500, batch_size=20):
        self.model_file = model_file
        self.prep_file = prep_file
        self.prep_k_file = prep_k_file
        self.v = v
        self.xx = xx
        self.num_angles = num_angles
        self.slice_resolution = slice_resolution
        self.batch_size = min(60, cpu_count() * 6)  # Auto-adjusts to your core count
        self.marked_point = None
        self.model = None
        self.model_weights_file = "tooth_segmentation_model.weights.h5"
        self.correction_data = []
        self.last_retrain_size = 0  # Track training data size
        self.angle_corrections = {}  # Track corrections per angle
        self.current_correction_id = 0  # Unique ID for corrections
        self.marked_umschlagpunkt = None
        self.mesh_processor = None
        self.is_flipped = False
                # --- 3D preview window state (vedo) ---
        self.plt_3d = None
        self.vmesh_model = None
        self.vmesh_prep = None
        self.vmesh_prep_k = None
        self.slice_plane = None

        
        # Initialize model and data
        self.initialize_model()
        self.load_and_process_meshes()
        self.precompute_slice_data()


    def initialize_model(self):
        """Improved model for spatial-angular learning"""
        input_shape = (self.slice_resolution, self.slice_resolution, 3)
        
        inputs = layers.Input(shape=input_shape)
        
        # Angular pathway
        angle_features = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
        angle_features = layers.MaxPooling2D((2, 2))(angle_features)
        
        # Spatial pathway
        spatial_features = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        spatial_features = layers.MaxPooling2D((2, 2))(spatial_features)
        
        # Ensure both pathways have the same spatial dimensions
        # If necessary, apply padding or cropping
        if spatial_features.shape[1] != angle_features.shape[1] or spatial_features.shape[2] != angle_features.shape[2]:
            # Apply padding to match dimensions
            spatial_features = layers.ZeroPadding2D(((0, 1), (0, 1)))(spatial_features)
        
        # Merge pathways
        merged = layers.Concatenate()([spatial_features, angle_features])
        merged = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
        merged = layers.GlobalAveragePooling2D()(merged)
        
        outputs = layers.Dense(2, activation='sigmoid')(merged)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def load_and_process_meshes(self):
        #print("Loading and processing meshes...")
        
        # Load meshes
        self.mesh = trimesh.load(self.model_file)  # VOP
        self.prep_mesh = trimesh.load(self.prep_file)  # Prep margin
        self.prep_k_mesh = trimesh.load(self.prep_k_file)  # Prep margin, conventional

        # Generate tooth ID from v and xx
        tooth_id = f"{self.v}{self.xx}"
        icp_filename = f"Zähne/{self.v}/{self.xx}/icp_points_{tooth_id}.npy"

        # Check if ICP points are saved
        if os.path.exists(icp_filename):
            #print(f"Loading ICP points from {icp_filename}...")
            icp_points = np.load(icp_filename, allow_pickle=True).item()
            source_points = icp_points["source_points"]
            target_points = icp_points["target_points"]

            self.is_flipped = icp_points.get("is_flipped", False)

            # Compute initial transformation using saved points
            initial_T = best_fit_transform(source_points, target_points)
            self.prep_k_mesh.apply_transform(initial_T)

            # Sample points from both meshes for ICP
            target_points_icp = self.prep_mesh.sample(5000)  # Points from präpgrenze (target)
            source_points_icp = self.prep_k_mesh.sample(5000)  # Points from präpgrenze_k (source)

            # Run custom ICP to align präpgrenze_k to präpgrenze
            T, finalA, final_error, iterations = iterative_closest_point(
                source_points_icp, 
                target_points_icp,
                max_iterations=500,  # Increase iterations for better alignment
                tolerance=1e-6       # Smaller tolerance for finer alignment
            )

            # Apply the ICP transformation to the entire präpgrenze_k mesh
            self.prep_k_mesh.apply_transform(T)
        else:
            # If no saved points, run interactive ICP
            print(f"No saved ICP points found for tooth {tooth_id}. Running interactive ICP...")
            self.prep_k_mesh = align_meshes_interactively(self.prep_mesh, self.prep_k_mesh, tooth_id, self.v, self.xx) 

        # Align meshes to the estimated tooth long axis before any rotation or slicing
        self.align_meshes_to_tooth_axis()
 
        # Calculate the shared bounding box that encompasses all meshes
        self.shared_bounds = np.array([
            np.minimum(
                np.minimum(self.mesh.bounds[0], self.prep_mesh.bounds[0]),
                self.prep_k_mesh.bounds[0]
            ),  # Min bounds
            np.maximum(
                np.maximum(self.mesh.bounds[1], self.prep_mesh.bounds[1]),
                self.prep_k_mesh.bounds[1]
            )   # Max bounds
        ])

        # === Add 15% padding ===
        bounds_size = self.shared_bounds[1] - self.shared_bounds[0]
        padding = bounds_size * 0.4
        self.shared_bounds[0] -= padding
        self.shared_bounds[1] += padding

        # === OPTIMIZATION 5: Initialize optimized processor ===
        self.mesh_processor = OptimizedMeshProcessor(self.shared_bounds, self.slice_resolution)
        # Match mm-per-pixel to the actual rasterized YZ bounds and scaling.
        self.mm_per_pixel = self.mesh_processor.y_range / (self.slice_resolution - 1)
        

    def align_meshes_to_tooth_axis(self):
        """Align meshes to the tooth long axis estimated from the prep_k mesh."""
        if self.mesh is None or self.prep_mesh is None or self.prep_k_mesh is None:
            return

        verts = np.asarray(self.prep_k_mesh.vertices, dtype=float)
        if verts.shape[0] < 3:
            return

        centroid = verts.mean(axis=0)

        axis = None
        axis_center = None
        try:
            faces = np.asarray(self.prep_k_mesh.faces, dtype=int)
            if faces.ndim == 2 and faces.shape[1] >= 3:
                edges = np.sort(
                    np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1,
                )
                unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
                boundary_edges = unique_edges[counts == 1]
            else:
                boundary_edges = np.zeros((0, 2), dtype=int)
        except Exception:
            boundary_edges = np.zeros((0, 2), dtype=int)

        if boundary_edges.shape[0] > 0:
            adjacency = {}
            for a, b in boundary_edges:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)

            visited = set()
            loops = []
            for v in adjacency:
                if v in visited:
                    continue
                stack = [v]
                component = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    component.append(cur)
                    for nbr in adjacency.get(cur, []):
                        if nbr not in visited:
                            stack.append(nbr)
                loops.append(component)

            if loops:
                loops.sort(key=len, reverse=True)
                loop_indices = loops[0]
                loop_points = verts[loop_indices]

                loop_centroid = loop_points.mean(axis=0)
                loop_centered = loop_points - loop_centroid
                try:
                    _, _, vh_loop = np.linalg.svd(loop_centered, full_matrices=False)
                    axis = vh_loop[-1]
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 0:
                        axis = axis / axis_norm
                except np.linalg.LinAlgError:
                    axis = None

                if axis is not None:
                    mesh_centroid = verts.mean(axis=0)
                    to_bulk = mesh_centroid - loop_centroid
                    if np.dot(axis, to_bulk) < 0.0:
                        axis = -axis
                    axis_center = np.array(self.prep_mesh.centroid, dtype=float)

        if axis is None:
            centered = verts - centroid

            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return

            axis = vh[0]
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                return
            axis = axis / axis_norm

            projections = centered @ axis
            min_idx = int(np.argmin(projections))
            max_idx = int(np.argmax(projections))
            direction = verts[max_idx] - verts[min_idx]
            if np.dot(axis, direction) < 0.0:
                axis = -axis

        target = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(axis, target), -1.0, 1.0))

        # Keep orientation neutral by choosing the minimal rotation direction.
        if dot < 0.0:
            axis = -axis
            dot = -dot

        cross = np.cross(axis, target)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-8:
            self.tooth_axis_center = np.array(self.prep_mesh.centroid, dtype=float)
            return

        rotation_axis = cross / cross_norm
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))

        prep_centroid = np.array(self.prep_mesh.centroid, dtype=float)
        if axis_center is None:
            axis_center = centroid + axis * np.dot((prep_centroid - centroid), axis)
        else:
            axis_center = axis_center + axis * np.dot((prep_centroid - axis_center), axis)
        self.tooth_axis_center = axis_center

        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle, rotation_axis, axis_center
        )

        self.mesh.apply_transform(rotation_matrix)
        self.prep_mesh.apply_transform(rotation_matrix)
        self.prep_k_mesh.apply_transform(rotation_matrix)


    def rotate_mesh_around_point(self, mesh, angle_degrees, center_point):
        """Rotate mesh around a specific point (matching voxel rotation behavior)"""
        angle_rad = np.deg2rad(angle_degrees)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle_rad, [0, 0, 1], center_point  # Rotate around Z-axis at shared center
        )
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(rotation_matrix)
        return rotated_mesh
        
    def get_plane_intersection(self, mesh, plane_origin, plane_normal):
        """Get intersection of mesh with plane"""
        try:
            # Use trimesh's built-in plane intersection
            intersection = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if intersection is None:
                return []
                
            # Convert to 2D coordinates in the plane
            # For X-plane (normal=[1,0,0]), we want Y-Z coordinates
            if hasattr(intersection, 'entities'):
                lines = []
                for entity in intersection.entities:
                    if hasattr(entity, 'points'):
                        # Get 3D points and project to 2D
                        points_3d = intersection.vertices[entity.points]
                        # For X-plane, use Y and Z coordinates
                        points_2d = points_3d[:, [1, 2]]  # Y, Z
                        lines.append(points_2d)
                return lines
            else:
                # Single path case
                if len(intersection.vertices) > 0:
                    points_2d = intersection.vertices[:, [1, 2]]  # Y, Z
                    return [points_2d]
                    
        except Exception as e:
            print(f"Intersection error: {e}")
            
        return []
        
    def draw_line(self, image, p1, p2):
        """Draw line between two points using Bresenham's algorithm"""
        row1, col1 = p1  # Note: using row/col terminology for clarity
        row2, col2 = p2
        
        # Bresenham's line algorithm
        drow = abs(row2 - row1)
        dcol = abs(col2 - col1)
        
        row_step = 1 if row1 < row2 else -1
        col_step = 1 if col1 < col2 else -1
        
        if dcol > drow:
            err = dcol / 2
            row = row1
            for col in range(col1, col2 + col_step, col_step):
                if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
                    image[row, col] = 1.0
                err -= drow
                if err < 0:
                    row += row_step
                    err += dcol
        else:
            err = drow / 2
            col = col1
            for row in range(row1, row2 + row_step, row_step):
                if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
                    image[row, col] = 1.0
                err -= dcol
                if err < 0:
                    col += col_step
                    err += drow

    def get_slice_at_angle(self, angle_degrees):
        """Get 2D slice at given rotation angle"""
        # Rotate all meshes around the abutment center (Step 4 requirement)
        rotation_center = self.prep_mesh.centroid
        
        # === OPTIMIZATION 7: Use cached rotation matrices ===
        rotation_matrix = self.mesh_processor.get_rotation_matrix(angle_degrees)
        
        tooth_rotated = self.rotate_mesh_around_point(self.mesh, angle_degrees, rotation_center)
        prep_rotated = self.rotate_mesh_around_point(self.prep_mesh, angle_degrees, rotation_center)
        prep_k_rotated = self.rotate_mesh_around_point(self.prep_k_mesh, angle_degrees, rotation_center)
        
        # Define cutting plane (X = center, like in voxel approach)
        center_x = rotation_center[0]
        plane_origin = np.array([center_x, 0, 0])
        plane_normal = np.array([1, 0, 0])
        
        # Get intersections
        tooth_lines = self.get_plane_intersection(tooth_rotated, plane_origin, plane_normal)
        prep_lines = self.get_plane_intersection(prep_rotated, plane_origin, plane_normal)
        prep_k_lines = self.get_plane_intersection(prep_k_rotated, plane_origin, plane_normal)
        
        
        # Rasterize to images
        tooth_image = self.mesh_processor.rasterize_lines_optimized(tooth_lines)
        prep_image = self.mesh_processor.rasterize_lines_optimized(prep_lines)
        prep_k_image = self.mesh_processor.rasterize_lines_optimized(prep_k_lines)
        
        return tooth_image, prep_image, prep_k_image

    def precompute_slice_data(self):
        """Precompute all slice data for all angles"""
        #print("Precomputing slice data for all angles...")
        self.angles = np.linspace(0, 360, self.num_angles, endpoint=False)
        
        self.slice_data_cache = [None] * self.num_angles
        self.prep_slice_data_cache = [None] * self.num_angles
        self.prep_k_slice_data_cache = [None] * self.num_angles
        
        for i, angle in tqdm(enumerate(self.angles), desc="Computing slices"):
            tooth_img, prep_img, prep_k_img = self.get_slice_at_angle(angle)
            self.slice_data_cache[i] = tooth_img
            self.prep_slice_data_cache[i] = prep_img
            self.prep_k_slice_data_cache[i] = prep_k_img

    def flip_model_180(self, event):
        """Flip the display by 180 degrees and save state"""
        #print("Flipping view by 180 degrees...")
        self.is_flipped = not self.is_flipped
        
        # Save the flip state to ICP file
        tooth_id = f"{self.v}{self.xx}"
        icp_filename = f"Zähne/{self.v}/{self.xx}/icp_points_{tooth_id}.npy"
        if os.path.exists(icp_filename):
            icp_points = np.load(icp_filename, allow_pickle=True).item()
            icp_points["is_flipped"] = self.is_flipped
            np.save(icp_filename, icp_points)
        
        # Update the display
        current_angle_idx = int(self.slider.val)
        self.plot_slice(current_angle_idx)
        
        print(f"View flipped {'(reversed)' if self.is_flipped else '(normal)'}")

    def plot_slice(self, angle_idx):
        angle = self.angles[angle_idx]
        
        # Get cached slice data
        slice_data_resized = self.slice_data_cache[angle_idx]
        prep_slice_data_resized = self.prep_slice_data_cache[angle_idx]
        prep_k_slice_data_resized = self.prep_k_slice_data_cache[angle_idx]

        if self.is_flipped:
            slice_data_resized = np.fliplr(np.flipud(slice_data_resized))
            prep_slice_data_resized = np.fliplr(np.flipud(prep_slice_data_resized))
            prep_k_slice_data_resized = np.fliplr(np.flipud(prep_k_slice_data_resized))

        self.ax.clear()
        from matplotlib.colors import ListedColormap
        # Base: inverted grayscale (0=white background, 1=black lines)
        base_cmap = 'gray_r'
        # Overlays: transparent for 0, solid color for 1 to avoid purple tints
        green_overlay = ListedColormap([(0, 0, 0, 0), (0, 0.8, 0, 1)])
        blue_overlay = ListedColormap([(0, 0, 0, 0), (0, 0.4, 1, 1)])

        self.ax.imshow(slice_data_resized, cmap=base_cmap, origin='lower', interpolation='none', vmin=0, vmax=1)
        self.ax.imshow(prep_slice_data_resized, cmap=green_overlay, origin='lower', interpolation='none', vmin=0, vmax=1)
        self.ax.imshow(prep_k_slice_data_resized, cmap=blue_overlay, origin='lower', interpolation='none', vmin=0, vmax=1)

        self.ax.set_title(f'Direct Mesh Intersection - Rotation angle: {angle:.2f}° around Z-axis, Slice along X-axis')

        # === Plot Center of Rotation ===
        center_3d = self.prep_mesh.centroid
        mp = self.mesh_processor
        # Y (3D) -> X (Plot), Z (3D) -> Y (Plot)
        center_pixel_y = (center_3d[1] - mp.bounds_2d[0][0]) * mp.y_scale
        center_pixel_z = (center_3d[2] - mp.bounds_2d[0][1]) * mp.z_scale
        
        if self.is_flipped:
            center_pixel_y = self.slice_resolution - 1 - center_pixel_y
            center_pixel_z = self.slice_resolution - 1 - center_pixel_z
            
        self.ax.plot(center_pixel_y, center_pixel_z, 'r+', markersize=15, markeredgewidth=2, label="Center of Rotation")

        # === Use unified calculation method ===
        final_mep, current_sulcusboden, current_vop, extended_line_points, sulcus_line, vop_line, sulcus_distance_mm, vop_distance_mm, sulcustiefe = \
            calculate_mep_and_detection_points(slice_data_resized, prep_slice_data_resized, prep_k_slice_data_resized, self.mm_per_pixel)

        präpgrenze_left = find_bottom_left(prep_slice_data_resized)
        präpgrenze_right = find_bottom_right(prep_slice_data_resized)
        präpgrenze_k_right = find_bottom_right(prep_k_slice_data_resized)

        if präpgrenze_left:
            self.ax.plot(präpgrenze_left[0], präpgrenze_left[1], color='orange', linestyle='None', marker='o', markersize=8, label=f'Margin (Dig., left) {präpgrenze_left[0]}, {präpgrenze_left[1]}')
        if präpgrenze_right:
            self.ax.plot(präpgrenze_right[0], präpgrenze_right[1], 'yo', markersize=8, label=f'Margin (Dig., right) {präpgrenze_right[0]}, {präpgrenze_right[1]}')
        if präpgrenze_k_right:
            self.ax.plot(präpgrenze_k_right[0], präpgrenze_k_right[1], 'co', markersize=8, label=f'Margin (Conv.) {präpgrenze_k_right[0]}, {präpgrenze_k_right[1]}')
        
        if sulcus_line:
            self.ax.plot([sulcus_line[0][0], sulcus_line[1][0]],
                        [sulcus_line[0][1], sulcus_line[1][1]],
                        'b--', linewidth=1, alpha=0.5, label="Sulcus floor Detection Line")
        
        if vop_line:
            self.ax.plot([vop_line[0][0], vop_line[1][0]],
                        [vop_line[0][1], vop_line[1][1]],
                        'g--', linewidth=1, alpha=0.5, label="VOP Detection Line")
            
        if extended_line_points:
            extended_left, extended_right = extended_line_points
            self.ax.plot([extended_left[0], extended_right[0]],
                        [extended_left[1], extended_right[1]],
                        'k-', linewidth=1, alpha=0.7, label="Margin Extrapolation Line (MEL)")
            
        
        if current_sulcusboden:
            if sulcus_distance_mm is not None:
                self.ax.plot(
                    current_sulcusboden[0], current_sulcusboden[1], 'bo',
                    markersize=8, alpha=0.7,
                    label=(
                        f'Sulcus floor ({current_sulcusboden[0]:.0f}, {current_sulcusboden[1]:.0f}); ' 
                        f'{sulcus_distance_mm:.2f} mm from MEL'
                    )
                )
            else:
                self.ax.plot(
                    current_sulcusboden[0], current_sulcusboden[1], 'bo',
                    markersize=8, alpha=0.7,
                    label=f'Sulcus floor ({current_sulcusboden[0]:.0f}, {current_sulcusboden[1]:.0f})'
                )

        if current_vop:
            if vop_distance_mm is not None:
                self.ax.plot(
                    current_vop[0], current_vop[1], 'go',
                    markersize=8, alpha=0.7,
                    label=f'VOP ({current_vop[0]:.0f}, {current_vop[1]:.0f}); {vop_distance_mm:.2f} mm from MEL'
                )
            else:
                self.ax.plot(
                    current_vop[0], current_vop[1], 'go',
                    markersize=8, alpha=0.7,
                    label=f'VOP ({current_vop[0]:.0f}, {current_vop[1]:.0f})'
                )

        # Plot mep (only if we found an intersection and distance <= 5mm)
        if final_mep and präpgrenze_right:
            distance_px = np.sqrt((final_mep[0] - präpgrenze_right[0])**2 +
                                (final_mep[1] - präpgrenze_right[1])**2)
            distance_mm = distance_px * self.mm_per_pixel
            
            if 0.1 <= distance_mm <= 3:    
                self.ax.plot(final_mep[0], final_mep[1], 'mo', 
                            markersize=8, alpha=0.7, 
                            label=f'Margin Extrapolation Point ({final_mep[0]:.0f}, {final_mep[1]:.0f}) - {distance_mm:.2f}mm')
                
                # Draw connecting line
                self.ax.plot([final_mep[0], präpgrenze_right[0]],
                            [final_mep[1], präpgrenze_right[1]],
                            'm-', linewidth=1, alpha=0.5)

        # ====== Update Legend ====== 
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                fontsize=9,
                frameon=True,
                framealpha=0.9,
                facecolor="#F6F4EF",
                edgecolor="#D9D2C5",
            )
        plt.draw()

    # Add this new helper method to the class
    def line_intersection(self, line1, line2):
        """Optimized line intersection wrapper"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        x, y = line_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4)
        return (x, y) if x is not None else None

    
    def clear_point(self, event):
        self.marked_point = None
        self.plot_slice(int(self.slider.val))


    def toggle_3d_preview(self, event=None):
        """Open/close the separate 3D window."""
        if self.plt_3d is not None:
            try:
                self.plt_3d.close()
            except Exception:
                pass

            self.plt_3d = None
            self.vmesh_model = None
            self.vmesh_prep = None
            self.vmesh_prep_k = None
            self.slice_plane = None
            return

        self.create_3d_viewer()
        # initialize plane to current slider value (or 0 if slider not created yet)
        idx = int(self.slider.val) if hasattr(self, "slider") and self.slider is not None else 0
        self.update_3d_slice_plane(idx)


    def create_3d_viewer(self):
        """Create a 3D viewer window showing meshes + the current slice plane."""
        print("[3D] Initializing Plotter...")
        self.plt_3d = Plotter(
            title="3D Slice Plane Preview",
            size=(900, 700),
            pos=(50, 50),
            bg="white",
            bg2="white",
        )

        # Use already-loaded meshes (aligned in load_and_process_meshes)
        model_tm = self.mesh
        prep_tm  = self.prep_mesh
        prep_k_tm = self.prep_k_mesh

        # Convert trimesh -> vedo Mesh
        self.vmesh_model = Mesh([model_tm.vertices, model_tm.faces]).c("green").alpha(0.35)
        self.vmesh_prep  = Mesh([prep_tm.vertices,  prep_tm.faces]).c("red").alpha(0.35)
        self.vmesh_prep_k = Mesh([prep_k_tm.vertices, prep_k_tm.faces]).c("blue").alpha(0.35)

        # Keep original meshes for clipping updates
        self.vmesh_model_src = self.vmesh_model.clone()
        self.vmesh_prep_src = self.vmesh_prep.clone()
        self.vmesh_prep_k_src = self.vmesh_prep_k.clone()

        self.plt_3d.add(self.vmesh_model, self.vmesh_prep, self.vmesh_prep_k)

        # Create initial plane
        try:
            self.slice_plane = self.create_slice_plane(angle_idx=0)
            if self.slice_plane is not None:
                self.plt_3d.add(self.slice_plane)
        except Exception as e:
            print("[3D] Slice plane creation failed:", e)
            self.slice_plane = None

        # IMPORTANT: show with explicit viewup/resetcam, then render once
        self.plt_3d.show(interactive=False, viewup="z", resetcam=True)

        try:
            # Some backends need an explicit first render
            self.plt_3d.render()
        except Exception:
            pass

        # Initialize clipping to current angle
        self.update_3d_slice_plane(0)



    def create_slice_plane(self, angle_idx: int):
        """Create a vedo Plane representing the current slice plane orientation."""
        angle_deg = float(self.angles[int(angle_idx)])
        angle_rad = np.deg2rad(angle_deg)

        # Your 2D slicing logic rotates meshes by +angle and slices with plane x = center_x.
        # Equivalent in world coords: keep meshes fixed and rotate the plane normal by -angle.
        rotation_center = np.array(self.prep_mesh.centroid, dtype=float)

        # Put the plane tile through x=center_x, but center it in Y/Z using the shared bounds
        yz_center = (self.shared_bounds[0, 1:3] + self.shared_bounds[1, 1:3]) / 2.0
        center = np.array([rotation_center[0], yz_center[0], yz_center[1]], dtype=float)


        # Base plane normal is +X (x = constant)
        base_normal = np.array([1.0, 0.0, 0.0], dtype=float)

        # Rotate normal around Z by -angle
        c = np.cos(-angle_rad)
        s = np.sin(-angle_rad)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)
        normal = Rz @ base_normal

        # Set plane size based on overall scene bounds
        all_verts = np.vstack([self.mesh.vertices, self.prep_mesh.vertices, self.prep_k_mesh.vertices])
        mins = all_verts.min(axis=0)
        maxs = all_verts.max(axis=0)
        extent = float(np.max(maxs - mins))
        size = max(extent * 1.2, 1.0)

        # vedo Plane expects pos and normal; sx/sy define plane dimensions
        pl = Plane(pos=tuple(center), normal=tuple(normal), sx=size, sy=size)
        pl.c("yellow").alpha(0.25)
        return pl


    def update_3d_slice_plane(self, angle_idx: int):
        """Update plane in the 3D preview to match current slider angle."""

        plt3d = getattr(self, "plt_3d", None)
        if plt3d is None:
            return

        # If the window was closed, renderer may be None
        if getattr(plt3d, "renderer", None) is None:
            self.plt_3d = None
            self.slice_plane = None
            return

        # Remove old plane
        old_plane = getattr(self, "slice_plane", None)
        if old_plane is not None:
            try:
                plt3d.remove(old_plane)
            except Exception:
                pass
            self.slice_plane = None

        # Add updated plane
        try:
            self.slice_plane = self.create_slice_plane(angle_idx)
            if self.slice_plane is not None:
                plt3d.add(self.slice_plane)
        except Exception as e:
            print("[3D] Plane update failed:", e)
            self.slice_plane = None
            return

        # Render refresh
        try:
            plt3d.render()
        except Exception:
            pass



    def create_slice_plane(self, angle_idx):
        """Create a plane representing the current slice view"""
        angle = self.angles[angle_idx]
        
        # Calculate center from the mesh vertices
        if hasattr(self, 'mesh_vedo'):
            vertices = self.mesh_vedo.points()
            center = np.mean(vertices, axis=0)
        else:
            center = [0, 0, 0]  # Fallback center
        
        # Create plane perpendicular to X-axis with appropriate size
        bounds_size = self.shared_bounds[1] - self.shared_bounds[0]
        plane_size = np.max(bounds_size) * 2
        
        plane = Plane(
            pos=center,
            normal=(1, 0, 0),
            s=(plane_size, plane_size)
        )
        
        # Rotate around Z axis to match 2D view
        plane.rotate(angle, axis=(0, 0, 1), point=center)
        
        # Style the plane with 20% opacity fill and wireframe
        plane.color('yellow').alpha(0.2).wireframe(False).lighting('ambient')
        return plane

    def update_3d_slice_plane(self, angle_idx):
        """Update the 3D slice plane to match current 2D view"""

        # 1) Guard: if 3D window is not open (or already closed), do nothing
        plt3d = getattr(self, "plt_3d", None)
        if plt3d is None:
            return

        # Some vedo backends may set renderer to None after closing
        if getattr(plt3d, "renderer", None) is None:
            self.plt_3d = None
            return

        # 2) Remove old plane if it exists
        old_plane = getattr(self, "slice_plane", None)
        if old_plane is not None:
            try:
                plt3d.remove(old_plane)
            except Exception:
                pass
            self.slice_plane = None

        # 3) Current angle
        angle = float(self.angles[int(angle_idx)])

        # 4) Plane center: use shared bounds center in Y/Z and keep X at slicing center
        # This makes the *finite* rectangle appear in the right place.
        b0 = self.shared_bounds[0]
        b1 = self.shared_bounds[1]
        yz_center = (b0[1:3] + b1[1:3]) / 2.0

        # Use your slicing center_x if you have it; otherwise fall back to prep centroid x
        if hasattr(self, "center_x"):
            cx = float(self.center_x)
        elif hasattr(self, "prep_mesh"):
            cx = float(self.prep_mesh.centroid[0])
        else:
            cx = float((b0[0] + b1[0]) / 2.0)

        center = (cx, float(yz_center[0]), float(yz_center[1]))

        # 5) Plane size from bounds
        bounds_size = (b1 - b0)
        plane_size = float(np.max(bounds_size) * 2.0)

        # 6) Create plane: base normal is +X (x = constant plane)
        self.slice_plane = Plane(
            pos=center,
            normal=(1, 0, 0),
            s=(plane_size, plane_size)
        )


        # 7) Rotate plane around Z to match 2D view.
        # If your 2D pipeline rotates meshes by +angle, the equivalent in world coords is rotating the plane by -angle.
        self.slice_plane.rotate(-angle, axis=(0, 0, 1), point=center)

        # 8) Styling
        self.slice_plane.color("yellow").alpha(0.2).wireframe(False).lighting("ambient")

        # 9) Add + render
        plt3d.add(self.slice_plane)

        # 10) Clip meshes to one side of the plane for preview
        base_normal = np.array([1.0, 0.0, 0.0], dtype=float)
        c = np.cos(np.deg2rad(-angle))
        s = np.sin(np.deg2rad(-angle))
        normal = np.array([c, s, 0.0], dtype=float)

        try:
            for attr_src, attr_dst in [
                ("vmesh_model_src", "vmesh_model"),
                ("vmesh_prep_src", "vmesh_prep"),
                ("vmesh_prep_k_src", "vmesh_prep_k"),
            ]:
                src = getattr(self, attr_src, None)
                dst = getattr(self, attr_dst, None)
                if src is None or dst is None:
                    continue
                try:
                    plt3d.remove(dst)
                except Exception:
                    pass
                clipped = src.clone().cut_with_plane(origin=center, normal=normal)
                setattr(self, attr_dst, clipped)
                plt3d.add(clipped)
        except Exception:
            pass

        try:
            plt3d.render()
        except Exception:
            pass

            
    def save_data(self, event):
        global all_data
        print("Saving data...")
        #csv_filename = f"Zähne/{self.v}/{self.xx}/output_{self.v}_{self.xx}.csv"
        csv_filename = f"Zähne/Output/{self.v}_{self.xx}.csv"
        csv_rows = []

        # Calculate the physical-to-pixel ratio
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["angle", "x", "y", "distance_mm"])
            for angle in range(0, 365, 5):
                angle_idx = min(angle, self.num_angles - 1)
                #print("angle: ", angle_idx)

                # Get the slice data for the current angle
                slice_data = self.slice_data_cache[angle_idx]
                prep_slice_data = self.prep_slice_data_cache[angle_idx]
                prep_k_slice_data = self.prep_k_slice_data_cache[angle_idx]

                # Apply flip if needed
                if self.is_flipped:
                    slice_data = np.fliplr(np.flipud(slice_data))
                    prep_slice_data = np.fliplr(np.flipud(prep_slice_data))
                    prep_k_slice_data = np.fliplr(np.flipud(prep_k_slice_data))

                # === Use unified calculation method ===
                final_mep, current_sulcusboden, current_vop, extended_line_points, sulcus_line, vop_line, sulcus_distance_mm, vop_distance_mm, sulcustiefe = \
                    calculate_mep_and_detection_points(slice_data, prep_slice_data, prep_k_slice_data, self.mm_per_pixel)


                # Use umschlagpunkt_1, präpgrenze bottom, and präpgrenze_k bottom
                präpgrenze_left = find_bottom_left(prep_slice_data)
                präpgrenze_point = find_bottom_right(prep_slice_data)
                präpgrenze_k_point = find_bottom_right(prep_k_slice_data)

                # Calculate the distance between präpgrenze and präpgrenze_k points in pixels
                MargDev_pixels = None
                MargDev_mm = None
                if präpgrenze_point and präpgrenze_k_point:
                    distance_px = np.sqrt((präpgrenze_k_point[0] - präpgrenze_point[0])**2 + 
                                        (präpgrenze_k_point[1] - präpgrenze_point[1])**2)
                    
                    y_diff = präpgrenze_point[1] - präpgrenze_k_point[1]
                    sign = 1 if y_diff >= 0 else -1
                    # Apply sign while preserving original distance magnitude
                    MargDev_pixels = sign * distance_px
                    MargDev_mm = MargDev_pixels * self.mm_per_pixel

                # Calculate distance from mep to Präpgrenze (viewer logic)
                SulcWid_pixels = None
                SulcWid_mm = None
                if final_mep and präpgrenze_point:
                    SulcWid_pixels = np.sqrt((final_mep[0] - präpgrenze_point[0])**2 +
                                        (final_mep[1] - präpgrenze_point[1])**2)
                    SulcWid_mm = SulcWid_pixels * self.mm_per_pixel
                    
                    # Match viewer filter: only keep within [0.1, 3] mm
                    if not (0.1 <= SulcWid_mm <= 3):
                        final_mep = None
                        SulcWid_pixels = None
                        SulcWid_mm = None
                
                
                # Prepare the row for the CSV
                row = {
                    "angle": angle,
                    "präpgrenze_x": präpgrenze_point[0] if präpgrenze_point else None,
                    "präpgrenze_y": präpgrenze_point[1] if präpgrenze_point else None,
                    "präpgrenze_k_x": präpgrenze_k_point[0] if präpgrenze_k_point else None,
                    "präpgrenze_k_y": präpgrenze_k_point[1] if präpgrenze_k_point else None,
                    "sulcusboden_x": current_sulcusboden[0] if current_sulcusboden else None,
                    "sulcusboden_y": current_sulcusboden[1] if current_sulcusboden else None,
                    "MargExPt_x": final_mep[0] if final_mep else None,
                    "MargExPt_y": final_mep[1] if final_mep else None,
                    "vop_x": current_vop[0] if current_vop else None,
                    "vop_y": current_vop[1] if current_vop else None,
                    #"MargDev_pixels": MargDev_pixels,  # Distance präpgrenze (d) to (k) in pixels
                    "MargDev": MargDev_mm, # Distance in millimeters
                    #"SulcWid": SulcWid_pixels, # Distance mep to Präpgrenze (d)
                    "SulcWid": SulcWid_mm, # in mm
                    "SulcDep": sulcustiefe,
                    "MargExPt_Sulcusboden": sulcus_distance_mm,
                    "vop_MargExPt": vop_distance_mm,
                }
                csv_rows.append(row)
        # Save the data to a CSV file
        with open(csv_filename, mode="w", newline="") as csv_file:
            fieldnames = [
                "angle", "präpgrenze_x", "präpgrenze_y", 
                "präpgrenze_k_x", "präpgrenze_k_y", "sulcusboden_x", "sulcusboden_y",
                "MargExPt_x", "MargExPt_y", "vop_x", "vop_y", "MargDev", "SulcWid",
                "SulcDep", "MargExPt_Sulcusboden", "vop_MargExPt"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"CSV data saved to {csv_filename}")

        # Save the current correction to all_data
        angle_idx = int(self.slider.val)
        slice_data = self.slice_data_cache[angle_idx]
        prep_slice_data = self.prep_slice_data_cache[angle_idx]
        prep_k_slice_data = self.prep_k_slice_data_cache[angle_idx]

        model_id = os.path.splitext(os.path.basename(self.model_file))[0]

        # Save training_data.npy
        print(f"Data for model '{model_id}' saved")
        
    def run(self):
        plt.rcParams.update({
            "figure.facecolor": "#F6F4EF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#C9C4B8",
            "axes.labelcolor": "#2B2B2B",
            "xtick.color": "#6C6C6C",
            "ytick.color": "#6C6C6C",
            "font.family": "Avenir",
            "font.size": 11,
        })

        self.fig, self.ax = plt.subplots(figsize=(9.5, 8.5))
        plt.subplots_adjust(left=0.08, right=0.78, bottom=0.30, top=0.92)
        try:
            self.fig.canvas.manager.set_window_title("DentalSlicer")
        except Exception:
            pass
        
        #self.ax.imshow(initial_slice, cmap='bone_r', origin='lower', interpolation='none')
        self.plot_slice(0)
        self.ax.set_title(f'Rotation angle: {self.angles[0]:.2f}° around Z-axis, Slice along X-axis')

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        ax_slider = plt.axes([0.18, 0.10, 0.64, 0.035], facecolor="#E9E5DB")
        self.slider = Slider(
            ax_slider,
            'Rotation Angle (Z)',
            0,
            self.num_angles - 1,
            valinit=0,
            valstep=1,
            color="#C8B9A8",
        )
        def _on_slider(val):
            idx = int(val)
            self.plot_slice(idx)
            if getattr(self, "plt_3d", None) is not None:
                self.update_3d_slice_plane(idx)

        self.slider.on_changed(_on_slider)

        ax_3d_button = plt.axes([0.18, 0.18, 0.18, 0.045])
        preview_3d_button = Button(ax_3d_button, '3D Preview', color="#EFEAE0", hovercolor="#E1D9CB")
        preview_3d_button.on_clicked(self.toggle_3d_preview)

        ax_save_button = plt.axes([0.74, 0.18, 0.12, 0.045])
        save_button = Button(ax_save_button, 'Save Data', color="#EFEAE0", hovercolor="#E1D9CB")
        save_button.on_clicked(self.save_data)

        ax_flip_button = plt.axes([0.88, 0.18, 0.1, 0.045])
        flip_button = Button(ax_flip_button, 'Flip 180°', color="#EFEAE0", hovercolor="#E1D9CB")
        flip_button.on_clicked(self.flip_model_180)

        self.fig.text(
            0.985,
            0.015,
            "© Majed Kutaini",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#9B9588",
        )

        # Connect mouse scroll event for zooming
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()
    
    def on_key_press(self, event):
        if event.key == '0':  # Reset zoom
            self.reset_zoom()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_mouse, y_mouse = event.xdata, event.ydata
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        new_xlim = (x_mouse - (x_mouse - xlim[0]) * zoom_factor,
                    x_mouse + (xlim[1] - x_mouse) * zoom_factor)
        new_ylim = (y_mouse - (y_mouse - ylim[0]) * zoom_factor,
                    y_mouse + (ylim[1] - y_mouse) * zoom_factor)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw()

    def reset_zoom(self):
        """Reset the zoom to the original axis limits."""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.fig.canvas.draw()
