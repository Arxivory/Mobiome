import pyvista as pv
import numpy as np

class Biomechanical3DViewer:
    """
    3D Interactive Analytics Viewer for Visualizing Skeletal Trajectories,
    Joint Torque Vector Arrows, and Dynamic Kinetic Heatmaps.
    """
    def __init__(self, p_3d: np.ndarray, tau: np.ndarray, fps: float = 60.0):
        """
        p_3d: (T, 17, 3) 3D Cartesian Keypoints in Meters
        tau:  (T, 4) Joint Torques in N·m
        """
        self.p_3d = p_3d
        self.tau = tau
        self.fps = fps
        self.T, self.K, _ = p_3d.shape

        # Standard H36M 17-Joint Skeleton Connections
        self.skeleton_bones = [
            (0, 1), (1, 2), (2, 3),      # Right Leg (Pelvis -> R_Hip -> R_Knee -> R_Ankle)
            (0, 4), (4, 5), (5, 6),      # Left Leg  (Pelvis -> L_Hip -> L_Knee -> L_Ankle)
            (0, 7), (7, 8), (8, 9), (9, 10), # Spine / Neck / Head
            (8, 11), (11, 12), (12, 13), # Left Arm  (Neck -> L_Shoulder -> L_Elbow -> L_Wrist)
            (8, 14), (14, 15), (15, 16)  # Right Arm (Neck -> R_Shoulder -> R_Elbow -> R_Wrist)
        ]

        # Key joint indices matching torque outputs [R-Knee, L-Knee, R-Elbow, L-Elbow]
        self.monitored_joints = [2, 5, 15, 12]

        # Initialize PyVista Plotter
        self.plotter = pv.Plotter(title="EdgePhysio3D - Biomechanical Inverse Dynamics Analytics")
        self.plotter.set_background("#1e1e24")

        # Actors
        self.joint_spheres = None
        self.bone_lines = []
        self.vector_glyph_actor = None
        
        self.current_frame = 0

    def _get_torque_heatmap_color(self, val: float, max_val: float = 60.0) -> list:
        """Maps torque magnitude to an RGB color gradient (Green -> Yellow -> Red)."""
        norm = np.clip(abs(val) / max_val, 0.0, 1.0)
        return [norm, 1.0 - norm, 0.0]

    def build_scene(self):
        """Builds initial 3D mesh actors and adds time slider UI controls."""
        # 1. Render Joint Nodes
        initial_points = self.p_3d[0]
        self.joint_cloud = pv.PolyData(initial_points)
        
        # Add scalar field for Heatmap overlay based on initial frame torques
        node_heatmaps = np.zeros(self.K)
        for idx, j_idx in enumerate(self.monitored_joints):
            node_heatmaps[j_idx] = abs(self.tau[0, idx])

        self.joint_cloud["Torque_Heatmap"] = node_heatmaps
        self._update_joint_spheres()

        # 2. Render Bone Connections
        for parent, child in self.skeleton_bones:
            p1 = initial_points[parent]
            p2 = initial_points[child]
            line = pv.Line(p1, p2)
            actor = self.plotter.add_mesh(line, color="#4cc9f0", line_width=4)
            self.bone_lines.append((parent, child, actor))

        # 3. Render 3D Torque Vector Glyphs (Arrows)
        self._update_torque_vectors(0)

        self.plotter.set_background('white')

        # 4. Interactive Time Control Slider
        self.plotter.add_slider_widget(
            callback=self.on_frame_change,
            rng=[0, self.T - 1],
            value=0,
            title="Time Step / Frame Control",
            pointa=(0.1, 0.08),
            pointb=(0.4, 0.08),
            fmt="%0.0f"
        )

        # Add Coordinate Grid Axes
        self.plotter.show_bounds(grid='front', location='outer', all_edges=True)
        self.plotter.add_camera_orientation_widget()

    def _update_torque_vectors(self, frame_idx: int):
        """Updates 3D vector arrows indicating joint torque magnitude & directional axes."""
        centers = []
        vectors = []
        magnitudes = []

        for idx, j_idx in enumerate(self.monitored_joints):
            pos = self.p_3d[frame_idx, j_idx]
            tau_val = self.tau[frame_idx, idx]

            # Vector points vertically along Z/Y rotational axis proportional to torque
            vec = np.array([0.0, 0.0, tau_val * 0.01]) # Normalized scale factor
            centers.append(pos)
            vectors.append(vec)
            magnitudes.append(abs(tau_val))

        vector_pd = pv.PolyData(np.array(centers))
        vector_pd["vectors"] = np.array(vectors)
        vector_pd["mag"] = np.array(magnitudes)

        arrows = vector_pd.glyph(orient="vectors", scale="mag", factor=0.01, geom=pv.Arrow())
        
        if self.vector_glyph_actor is not None:
            self.plotter.remove_actor(self.vector_glyph_actor)
            
        self.vector_glyph_actor = self.plotter.add_mesh(arrows, color="#f72585", opacity=0.85)

    def _update_joint_spheres(self):
        """Rebuild the glyph mesh after joint positions or heatmaps change."""
        glyphs = self.joint_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.016))
        if self.joint_spheres is not None:
            self.plotter.remove_actor(self.joint_spheres)
        self.joint_spheres = self.plotter.add_mesh(
            glyphs,
            cmap="jet",
            clim=[0, 80],
            scalar_bar_args={"title": "Joint Torque Magnitude (N·m)"}
        )

    def on_frame_change(self, value):
        """Callback triggered on time-slider movement for real-time analysis."""
        frame_idx = int(value)
        self.current_frame = frame_idx
        pts = self.p_3d[frame_idx]

        # Update Spheres & Heatmaps
        self.joint_cloud.points = pts
        node_heatmaps = np.zeros(self.K)
        for idx, j_idx in enumerate(self.monitored_joints):
            node_heatmaps[j_idx] = abs(self.tau[frame_idx, idx])
        self.joint_cloud["Torque_Heatmap"] = node_heatmaps
        self._update_joint_spheres()

        # Update Bone Lines
        for parent, child, actor in self.bone_lines:
            p1 = pts[parent]
            p2 = pts[child]
            actor.mapper.dataset.points = np.array([p1, p2])

        # Update Vector Glyphs
        self._update_torque_vectors(frame_idx)
        self.plotter.render()

    def show(self):
        self.build_scene()
        self.plotter.show()