import numpy as np

COLOR_MAP = {
    "profile": "blue",
    "career": "green",
    "education": "#003366",
    "skills": "yellow",
    "certifications": "purple",
    "test_scores": "#3D005E",
    "honors": "#8B8000",
    "languages": "#2E0854",
    "volunteering": "#145A32",
    "projects": "orange",
    "publications": "#660000",
    "job_search": "#2F4F4F",
}


def extract_metadata(collection):
    """
    Extract vectors, documents, doc_types, and color mappings from a Chroma collection.

    Args:
        collection: The Chroma vectorstore collection object.

    Returns:
        tuple:
            - vectors (np.ndarray): Array of vector embeddings.
            - documents (list[str]): List of raw document texts.
            - doc_types (list[str]): List of document type labels (from metadata).
            - colors (list[str]): List of color codes mapped to each doc_type.
    """

    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    metadata = result["metadatas"]
    doc_types = [m.get("doc_type", "unknown") for m in metadata]
    colors = [COLOR_MAP.get(doc_type, "gray") for doc_type in doc_types]

    return vectors, documents, doc_types, colors
