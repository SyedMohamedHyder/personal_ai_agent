import numpy as np

# Color mapping for different document types in the LinkedIn knowledge base
# Each category gets a distinct color for visualization purposes
COLOR_MAP = {
    "profile": "#1e3a8a",
    "experience": "#065f46",
    "education": "#7c2d12",
    "skills": "#ca8a04",
    "certifications": "#7c3aed",
    "projects": "#ea580c",
    "publications": "#dc2626",
    "networking": "#0d9488",
    "communications": "#059669",
    "preferences": "#6b7280",
}


def extract_metadata(collection):
    """
    Extract vectors, documents, doc_types, and color mappings from a Chroma collection.
    
    The doc_types correspond to the organized LinkedIn directory structure:
    - profile: Basic profile information and summaries
    - experience: Work history, positions, volunteering
    - education: Educational background, honors, test scores
    - skills: Technical skills, languages, competencies
    - certifications: Professional certifications and credentials
    - projects: Portfolio projects and work samples
    - publications: Published articles and papers
    - networking: Connections, company follows, invitations
    - communications: Messages, coach interactions
    - preferences: Account settings, preferences, saved alerts

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
