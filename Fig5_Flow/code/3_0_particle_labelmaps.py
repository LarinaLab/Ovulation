import os

import numpy as np
import numpy.typing as npt
import nrrd
from scipy import ndimage
from skimage.segmentation import watershed
from PIL import Image


def find_isolated_groups(graph: dict[int, list[int]]) -> list[list[int]]:
    groups = []  # Isolated groups of nodes
    visited = set()  # Visited nodes
    
    # Helper function to perform Depth-First Search
    def dfs(node: int, group: list[int]) -> None:
        group.append(node)
        visited.add(node)
        # Explore neighbors
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, group)
    
    # Explore each node in the graph
    for node in graph:
        if node not in visited:
            # Start a new group
            group = []
            dfs(node, group)
            groups.append(group)
    
    return groups


dir_out = "labelmaps"
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

labels = [
    "Vivo_Control_hCG14h_040824",
    "Vivo_Control_hCG14h_041524",
    "Vivo_Control_hCG14h_041724",
    "Vivo_Padrin_hCG14h_041024",
    "Vivo_Padrin_hCG14h_041524",
    "Vivo_Padrin_hCG14h_041724",
]

for la in labels:
    print(la)

    img_path = f"particles/{la}_particles.nrrd"
    threshold = 20
    if not os.path.isfile(img_path):
        img_path = f"discriminator/eval/{la}_discriminated.nrrd"
        threshold = 99

    img, header = nrrd.read(img_path, index_order="C")
    img = img[4:-4, 4:-4, 4:-4]

    q = img > threshold
    space_only_element = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
    ])
    print("Labeling and finding particles...")
    labels, n_labels = ndimage.label(q, structure=space_only_element)
    obj = ndimage.find_objects(labels)
    lost_labels = np.zeros((n_labels,), bool)

    print("Eliminating small bits of junk...")
    for i in range(n_labels):
        patch = labels[obj[i]]
        mask = patch == i + 1
        if mask.sum() <= 2:
            patch[mask] = 0

    print("Merging and splitting particles...")
    terminates = 0
    grows = 0
    splits = 0
    retrosplits = 0
    excludes = 0

    NULL = {0}
    retro_groups = []
    for it in range(1, img.shape[0]):
        if it % 100 == 0:
            print(it)
        im_a = labels[it-1, None, :, :]
        im_b = labels[it, None, :, :]

        label_offset = im_a[im_a > 0].min()
        im_a_offset = im_a.copy()
        im_b_offset = im_b.copy()
        im_a_offset[im_a_offset > 0] -= label_offset - 1
        im_b_offset[im_b_offset > 0] -= label_offset - 1
        label_max = im_b_offset.max()
        objs_a = ndimage.find_objects(im_a_offset, label_max)
        objs_b = ndimage.find_objects(im_b_offset, label_max)

        # Create the particle connectivity graph for A and B.
        graph = dict()
        labels_a = set()  # Track these separately for distinguishing A from B.
        for ia in range(label_max):
            oa = objs_a[ia]
            if oa is None:
                continue
            label_a = ia + label_offset
            mask_a = im_a[oa] == label_a
            overlap_a2b = set(im_b[oa][mask_a]) - NULL
            labels_a.add(label_a)
            if label_a not in graph:
                graph[label_a] = []
            for label_b in overlap_a2b:
                # Establish a bidirectional graph connection.
                if label_b not in graph:
                    graph[label_b] = []
                graph[label_a].append(label_b)
                graph[label_b].append(label_a)

        groups = find_isolated_groups(graph)
        for g in groups:
            if len(g) <= 1:  # Case 0 - Terminate
                terminates += 1
            elif len(g) == 2:  # Case 1 - Grow
                label_a, label_b = g
                if label_b in labels_a:
                    label_a, label_b = label_b, label_a
                ib = label_b - label_offset
                im_b_ob = im_b[objs_b[ib]]
                im_b_ob[im_b_ob == label_b] = label_a
                grows += 1
            else:
                # How many A's there are in this group?
                is_in_a = [x in labels_a for x in g]
                na = sum(is_in_a)
                if len(g) - na == 1:  # Case 2 - Split B
                    labels_x = []
                    for q, x in zip(is_in_a, g):
                        if q:
                            labels_x.append(x)
                        else:
                            label_b = x
                    ib = label_b - label_offset
                    ob = objs_b[ib]
                    im_b_ob = im_b[ob]
                    mask = im_b_ob == label_b
                    topo = 255 - img[it, None, ob[1], ob[2]]
                    ws = watershed(topo, markers=im_a[ob], mask=mask)
                    im_b_ob[mask] = ws[mask]
                    splits += 1
                elif na == 1:  # Case 3 - Retro Split A
                    # These are tricky to do here efficiently and cannot affect
                    # future operations, so we record them for later.
                    labels_x = []
                    for q, x in zip(is_in_a, g):
                        if q:
                            label_a = x
                        else:
                            labels_x.append(x)
                    retro_groups.append((label_a, labels_x))
                    retrosplits += 1
                else:  # Case 4 - Too complicated to untangle.
                    excludes += 1

    # Apply the retro splits previously recorded.
    # These objects are 3D instead of the slices from earlier.
    print("Retro-splitting particles...")
    objs = ndimage.find_objects(labels)
    for label_a, labels_x in retro_groups:
        ia = label_a - 1
        oa = objs[ia]
        im_a_oa = labels[oa]
        mask = im_a_oa == label_a
        topo = 255 - img[oa]
        field = np.zeros(topo.shape, np.int32)
        field[-1, :, :] = labels[oa[0].stop, oa[1], oa[2]]
        ws = watershed(topo, markers=field, mask=mask)
        im_a_oa[mask] = ws[mask]

    print("Saving labels.")
    nrrd.write(f"{dir_out}/{la}_labels.nrrd", labels, header=header, index_order="C")
    labels_256 = (labels % 256).astype(np.uint8)
    nrrd.write(f"{dir_out}/{la}_labels256.nrrd", labels_256, header=header, index_order="C")
