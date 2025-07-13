from sklearn.manifold import TSNE
import plotly.graph_objects as go


def visualize_vectorstore_2d(vectors, documents, doc_types, colors, title="2D Chroma Vector Store Visualization"):
    """
    Reduce vector embeddings to 2D using t-SNE and return a Plotly figure.

    Args:
        vectors (ndarray): The high-dimensional vector embeddings.
        documents (list): The original text documents (same order as vectors).
        doc_types (list): The type of each document, used for hover info and coloring.
        colors (list): Color values corresponding to each doc_type.
        title (str): Title for the Plotly figure.

    Returns:
        go.Figure: A Plotly figure object ready to be displayed.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    hover_texts = [f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)]

    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=hover_texts,
        hoverinfo='text'
    )])

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='y',
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig


def visualize_vectorstore_3d(vectors, documents, doc_types, colors, title="3D Chroma Vector Store Visualization"):
    """
    Reduce vector embeddings to 3D using t-SNE and return a Plotly 3D scatter plot.

    Args:
        vectors (ndarray): High-dimensional vector embeddings.
        documents (list): Corresponding document texts.
        doc_types (list): Document types for labeling and coloring.
        colors (list): Color values for each point based on doc_type.
        title (str): Title for the Plotly figure.

    Returns:
        go.Figure: A Plotly 3D scatter plot figure object.
    """
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    hover_texts = [f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)]

    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=hover_texts,
        hoverinfo='text'
    )])

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig
