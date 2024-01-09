import numpy as np


def calculate_angles_with_axes(vectors):

    unit_vectors = {
    'X-axis': np.array([1, 0, 0]),
    'Y-axis': np.array([0, 1, 0]),
    'Z-axis': np.array([0, 0, 1])
    }
    
    angles = {axis: [] for axis in unit_vectors}  # Initialize as a dictionary of lists

    for vector in vectors:
        vector = np.array(vector)
        vector_magnitude = np.linalg.norm(vector)

        # Check for zero magnitude to avoid division by zero
        if vector_magnitude == 0:
            for axis in unit_vectors:
                angles[axis].append(0)
            continue

        for axis, unit_vector in unit_vectors.items():
            dot_product = np.dot(vector, unit_vector)
            # Clamp the value to the valid range for arccos to avoid numerical errors
            cos_angle = np.clip(dot_product / vector_magnitude, -1, 1)
            angle_rad = np.arccos(cos_angle)
            angles[axis].append(np.degrees(angle_rad))
    
    return angles

def calculate_vectors(locatios: dict):
    MC = locatios['chest'].values
    MS = locatios['shoulder'].values
    ME = locatios['elbow'].values
    MW = locatios['wrist'].values
    CtoS = MS - MC
    StoE = ME - MS
    EtoW = MW - ME
    return CtoS, StoE, EtoW


# if __name__=="__main__":

