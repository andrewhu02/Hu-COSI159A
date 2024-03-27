from pathlib import Path
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
import argparse
from model import slic, segments_to_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Simpler Linear Iterative Clustering")

    parser.add_argument('--image', type=str, help="Path to the input image", required=True)
    parser.add_argument('--output', type=str, help="Path to save the output image. Defaults to the same directory.", default="")
    parser.add_argument('--clusters', type=int, help="Number of clusters", default=250)
    parser.add_argument('--compactness', type=float, help="SLIC compactness parameter", default=20)
    parser.add_argument('--iterations', type=int, help="Number of iterations to run", default=10)
    parser.add_argument('--connected_components', type=bool, help="Whether to enforce connectivity of the segments", default=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = args.output if args.output \
        else image_path.parent / f"segmented_{args.clusters}_{args.compactness}_{image_path.name}.png"
    image = imread(image_path)
    
    segments = slic(image, compactness=args.compactness, num_clusters=args.clusters,
                     num_iterations=args.iterations, enforce_connectivity=args.connected_components)

    segmented_image = segments_to_image(image, segments)
    imsave(output_path, segmented_image)

    # Display the segmented image
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()
