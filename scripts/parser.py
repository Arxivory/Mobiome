import os
import glob
import numpy as np
import scipy.io as sio
import h5py

class H36MKeypointLoader:
    def __init__(self, target_joints: int = 17):
        """
        Human3.6M raw mDN datasets typically use 32 keypoints (or 17 selected main joints).
        Standard joint mapping for 17 skeleton points:
        [Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle, Spine, Neck, Head,
         Site, L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
        """
        self.target_joints = target_joints

    def load_from_npz(self, npz_file_path: str) -> np.ndarray:
        """Parses preprocessed NumPy .npz archives."""
        data = np.load(npz_file_path, allow_pickle=True)
        if 'positions_3d' in data:
            keypoints = data['positions_3d']
        elif 'keypoints3d' in data:
            keypoints = data['keypoints3d']
        else:
            first_key = list(data.keys())[0]
            keypoints = data[first_key]

        return self._format_tensor(keypoints)

    def _format_tensor(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Reshapes and converts keypoint matrices into standardized (T, 17, 3) shape in meters.
        """
        data = np.squeeze(np.array(raw_data))

        # Case 1: (T, 96) continuous flattened coordinates (32 joints * 3)
        if data.ndim == 2 and data.shape[1] in [96, 51]:
            T = data.shape[0]
            K = data.shape[1] // 3
            data = data.reshape(T, K, 3)

        # Case 2: (32, 3, T) or (3, 32, T) Matlab layout
        elif data.ndim == 3 and data.shape[2] > 200:
            if data.shape[0] in [17, 32]:
                data = np.transpose(data, (2, 0, 1))  # (T, K, 3)
            elif data.shape[1] in [17, 32]:
                data = np.transpose(data, (2, 1, 0))  # (T, K, 3)

        # Slice to standard 17-joint representation if raw keypoints use full 32 skeleton nodes
        if data.shape[1] == 32:
            h36m_17_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
            data = data[:, h36m_17_indices, :]

        # Scale millimeter annotations (e.g., 1800mm -> 1.8m) for biomechanical dynamics
        if np.max(np.abs(data)) > 100.0:
            data = data / 1000.0

        return data.astype(np.float32)