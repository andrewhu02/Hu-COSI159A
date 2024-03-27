import numpy as np
from numpy.linalg import norm
from skimage.color import rgb2lab

#Implementation based on the paper "SLIC Superpixels Compared to State-of-the-art Superpixel Methods" by Achanta et al.
def slic(image: np.ndarray, compactness: int, num_clusters: int,
            num_iterations: int = 10, enforce_connectivity: bool = True) -> np.ndarray:
    """Performs SLIC superpixel clustering on an image
    """

    # Convert to CIELAB space
    height = image.shape[0]
    width = image.shape[1]
    image_lab = rgb2lab(image)
    image_xy = np.argwhere(np.ones(image_lab.shape[:2])).reshape(height, width, 2)
    image_labxy = np.concatenate([image_lab, image_xy], axis=2)

    # Initialize cluster centers
    grid_interval = int(np.sqrt(height * width / num_clusters))
    center_positions = np.array([[i, j] for i in range(grid_interval // 2, height, grid_interval)
                                 for j in range(grid_interval // 2, width, grid_interval)])

    # Move each center to the closest adjacent pixel in 3x3
    padded_lab = np.pad(image_lab, ((2, 2), (2, 2), (0, 0)), constant_values=np.mean(image_lab))
    # Using 2 pixels of padding on each side so edges don't return NaN
    for idx, center in enumerate(center_positions + (2, 2)):
        i, j = center
        min_gradient = np.inf
        best_position = (i, j)
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                gradient = (norm(padded_lab[x + 1, y] - padded_lab[x - 1, y]) +
                            norm(padded_lab[x, y + 1] - padded_lab[x, y - 1]))
                if gradient < min_gradient:
                    min_gradient = gradient
                    best_position = (x, y)
        center_positions[idx] = best_position

    centers = np.array([[*image_lab[x, y], x, y] for x, y in center_positions])

    segments = np.zeros(image_lab.shape[:2], dtype=int)
    distances = np.full(image_lab.shape[:2], np.inf)

    # Iterate
    for _ in range(num_iterations):
        # Assign pixels to clusters based on color and spatial distances
        for idx, center in enumerate(centers):
            cl, ca, cb, cx, cy = center
            cx, cy = int(cx), int(cy)

            x_range = range(max(0, cx - grid_interval), min(height, cx + grid_interval))
            y_range = range(max(0, cy - grid_interval), min(width, cy + grid_interval))
            xx, yy = np.meshgrid(x_range, y_range)
            center_window = image_lab[xx, yy]

            d_color = np.sum((center_window - [cl, ca, cb]) ** 2, axis=2)
            d_space = (xx - cx) ** 2 + (yy - cy) ** 2
            d = np.sqrt(d_color + (d_space / grid_interval ** 2) * (compactness ** 2))

            # Update pixels closest to center window
            update_indices = np.where(d < distances[xx, yy])
            distances[xx[update_indices], yy[update_indices]] = d[update_indices]
            segments[xx[update_indices], yy[update_indices]] = idx

        # Update location of cluster centers
        for idx in range(len(centers)):
            segment_mask = segments == idx
            segment_labxy = image_labxy[segment_mask]
            segment_mean = np.mean(segment_labxy, axis=0)
            centers[idx] = segment_mean

    # Enforce connectivity of segments
    if enforce_connectivity:
        segments = enforce_connectivity_algorithm(segments, num_clusters)

    return segments


def enforce_connectivity_algorithm(segments: np.ndarray, num_clusters: int) -> np.ndarray:
    """Enforce connectivity algorithm as described in the original paper
    """

    height = segments.shape[0]
    width = segments.shape[1]
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    min_segment_size = (height * width) // num_clusters / 4  # 4 is a number from the paper

    new_clusters = np.full_like(segments, fill_value=-1, dtype=int)

    segment_label = 0
    for i in range(height):
        for j in range(width):
            if new_clusters[i, j] > -1:
                continue

            # Find all pixels connected to this pixel via breadth-first search
            connected_segment = []
            next_pixels = [(i, j)]
            while len(next_pixels):
                px, py = next_pixels.pop(0)
                connected_segment.append((px, py))
                for dx, dy in neighbors:
                    x, y = px + dx, py + dy
                    if 0 <= x < height and 0 <= y < width and new_clusters[x, y] == -1 and segments[i, j] == segments[x, y]:
                        next_pixels.append((x, y))
                        new_clusters[x, y] = segment_label

            # Merge small adjacent segments
            if len(connected_segment) < min_segment_size:
                adj_label = segment_label + 1
                for dx, dy in neighbors:
                    x, y = i + dx, j + dy
                    if 0 <= x < height and 0 <= y < width and new_clusters[x, y] >= 0 and new_clusters[x, y] != segment_label:
                        adj_label = new_clusters[x, y]

                # Reassign the pixels
                for x, y in connected_segment:
                    new_clusters[x, y] = adj_label
            else:
                # Increment the current segment label if the segment wasn't merged
                segment_label += 1

    return new_clusters


def segments_to_image(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Generate superpixel image
    """

    segmented_image = np.zeros_like(image)
    for segment_label in np.unique(segments):
        segment_mask = segments == segment_label
        segment_mean_color = np.mean(image[segment_mask], axis=0)
        segmented_image[segment_mask] = segment_mean_color
    return segmented_image