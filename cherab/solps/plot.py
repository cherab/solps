import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def plot_quadrangle_mesh(mesh, solps_data=None, ax=None):
    """
    Plots the quadrangle mesh grid geometry to a matplotlib figure.

    If solps_data is provided, it is used to colour the faces of the quadrangles in the mesh.
    If matplotlib axes are provided the collection is added to them them,
    otherwise a new figure and axes are created.

    :param mesh: SOLPSMesh object
    :param solps_data: Data array defined on the SOLPS mesh (optional)
    :param ax: matplotlib axes (optional)
    :return: matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    if solps_data is None:
        collection_mesh = create_quadrangle_polycollection(mesh, facecolor="none", edgecolor='b', linewidth=0.5)
    else:
        collection_mesh = create_quadrangle_polycollection(mesh, solps_data)
    ax.add_collection(collection_mesh)

    ax = _format_matplotlib_axes(ax, mesh)
    return ax


def plot_triangle_mesh(mesh, solps_data=None, ax=None):
    """
    Plots the triangle mesh grid geometry to a matplotlib figure.

    If solps_data is provided, it is used to colour the faces of the triangles in the mesh.
    If matplotlib axes are provided the collection is added to them them, 
    otherwise a new figure and axes are created.

    :param mesh: SOLPSMesh object
    :param solps_data: Data array defined on the SOLPS mesh
    :param ax: matplotlib axes (optional)
    :return: matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    if solps_data is None:
        collection_mesh = create_triangle_polycollection(mesh, facecolor="none", edgecolor='b', linewidth=0.5)
    else:
        collection_mesh = create_triangle_polycollection(mesh, solps_data)
    ax.add_collection(collection_mesh)

    ax = _format_matplotlib_axes(ax, mesh)
    return ax


def create_quadrangle_polycollection(mesh, solps_data=None, **collection_kw):
    """
    Creates a matplotlib PolyCollection object from the quadrangle mesh.

    If solps_data is provided, it is used to colour the faces of the quadrangles in the mesh.

    :param mesh: SOLPSMesh object
    :param solps_data: Data array defined on the SOLPS mesh
    :param collection_kw: Keyword arguments for the PolyCollection
    :return: matplotlib.collections.PolyCollection
    """
    verts = mesh.vertex_coordinates[mesh.quadrangles]
    collection_mesh = PolyCollection(verts, **collection_kw)
    if solps_data is not None:
        collection_mesh.set_array(solps_data[mesh.quadrangle_to_grid_map[:, 0], mesh.quadrangle_to_grid_map[:, 1]])
    return collection_mesh


def create_triangle_polycollection(mesh, solps_data=None, **collection_kw):
    """
    Creates a matplotlib PolyCollection object from the triangle mesh.

    If solps_data is provided, it is used to colour the faces of the triangles in the mesh.

    :param mesh: SOLPSMesh object
    :param solps_data: Data array defined on the SOLPS mesh
    :param collection_kw: Keyword arguments for the PolyCollection
    :return: matplotlib.collections.PolyCollection
    """
    verts = mesh.vertex_coordinates[mesh.triangles]
    collection_mesh = PolyCollection(verts, **collection_kw)
    if solps_data is not None:
        collection_mesh.set_array(solps_data[mesh.triangle_to_grid_map[:, 0], mesh.triangle_to_grid_map[:, 1]])
    return collection_mesh


def _format_matplotlib_axes(ax, mesh=None):
    """
    Formats the matplotlib axes for a SOLPS mesh plot.

    Sets aspect and labels for the axes. 
    If a SOLPSMesh object is provided, sets the limits of the axes to the mesh extent.

    :param ax: matplotlib axes
    :param mesh: SOLPSMesh object (optional)
    :return: matplotlib axes
    """
    ax.set_aspect(1)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")
    if mesh is not None:
        ax.set_xlim(mesh.mesh_extent["minr"], mesh.mesh_extent["maxr"])
        ax.set_ylim(mesh.mesh_extent["minz"], mesh.mesh_extent["maxz"])
    return ax
