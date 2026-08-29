import numpy as np

def similitud_coseno(vector_a, vector_b) -> float:
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)

    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)

    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norma_a * norma_b))