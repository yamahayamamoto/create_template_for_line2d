import numpy as np


def generate_shpere_surface_points_by_fib(N):
    f = (np.sqrt(5) - 1) / 2
    arr = np.linspace(-N, N, N * 2 + 1)
    theta = np.arcsin(arr / N)
    phi = 2 * np.pi * arr * f
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    points = np.array([x,y,z]).T
    return points


def hinter_sampling(min_n_points, radius=1.0):
    """Samples 3D points on a sphere surface by refining an icosahedron, as in:
    Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
    Local Patches with a Simple Linear Classifier, BMVC 2008

    :param min_n_points: The minimum number of points to sample on the whole sphere.
    :param radius: Radius of the sphere.
    :return: 3D points on the sphere surface and a list with indices of refinement
        levels on which the points were created.
    """
    # Vertices and faces of an icosahedron.
    a, b, c = 0.0, 1.0, (1.0 + np.sqrt(5.0)) / 2.0
    points = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
            (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b),
            (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
            (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
            (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
            (8, 6, 7), (9, 8, 1)]

    # Refinement levels on which the points were created.
    points_level = [0 for _ in range(len(points))]

    ref_level = 0
    while len(points) < min_n_points:
        ref_level += 1
        edge_pt_map = {}  # Mapping from an edge to a newly added point on the edge.
        faces_new = []  # New set of faces.

        # Each face is replaced by four new smaller faces.
        for face in faces:
            pt_inds = list(face)  # List of point ID's involved in the new faces.
            for i in range(3):

                # Add a new point if this edge has not been processed yet, or get ID of
                # the already added point.
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(points)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)

                    pt_new = 0.5 * (np.array(points[edge[0]]) + np.array(points[edge[1]]))
                    points.append(pt_new.tolist())
                    points_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])

            # Replace the current face with four new faces.
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                            (pt_inds[3], pt_inds[1], pt_inds[4]),
                            (pt_inds[3], pt_inds[4], pt_inds[5]),
                            (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new

    # Project the points to a sphere.
    points = np.array(points)
    points *= np.reshape(radius / np.linalg.norm(points, axis=1), (points.shape[0], 1))

    # Collect point connections.
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])

    # Order the points - starting from the top one and adding the connected points
    # sorted by azimuth.
    top_pt_id = np.argmax(points[:, 2])
    points_ordered = []
    points_todo = [top_pt_id]
    points_done = [False for _ in range(points.shape[0])]

    def calc_azimuth(x, y):
        two_pi = 2.0 * np.pi
        return (np.arctan2(y, x) + two_pi) % two_pi

    while len(points_ordered) != points.shape[0]:
        # Sort by azimuth.
        points_todo = sorted(points_todo, key=lambda i: calc_azimuth(points[i][0], points[i][1]))
        points_todo_new = []
        for pt_id in points_todo:
            points_ordered.append(pt_id)
            points_done[pt_id] = True
            points_todo_new += [i for i in pt_conns[pt_id]]  # Find the connected points.

        # Points to be processed in the next iteration.
        points_todo = [i for i in set(points_todo_new) if not points_done[i]]

    # Re-order the points and faces.
    points = points[np.array(points_ordered), :]
    points_level = [points_level[i] for i in points_ordered]
    points_order = np.zeros((points.shape[0],))
    points_order[np.array(points_ordered)] = np.arange(points.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [points_order[i] for i in faces[face_id]]
    # import open3d as o3d
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(points)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # o3d.io.write_triangle_mesh("./output/_hinter_sampling.ply", mesh, write_ascii=True)

    return points, points_level