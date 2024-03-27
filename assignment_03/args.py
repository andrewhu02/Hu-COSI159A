import argparse

parser = argparse.ArgumentParser(prog="Simpler Linear Iterative Clustering")

parser.add_argument('--image', type=str, help="Path to the input image", required=True)
parser.add_argument('--output', type=str, help="Path to save the output image. If not provided, will save in the same "
                                               "directory as the input image.", default="")
parser.add_argument('--algorithm', type=str, help="Algorithm to use; can be 'myslic' or 'scikit'", default='myslic')
parser.add_argument('--clusters', type=int, help="Number of clusters to form", default=250)
parser.add_argument('--compactness', type=float, help="SLIC compactness parameter", default=20)
parser.add_argument('--iterations', type=int, help="Number of iterations to run", default=10)
parser.add_argument('--connected_components', type=bool, help="Whether to enforce connectivity of the segments", default=True)
args = parser.parse_args()

image: str = args.image
output: str = args.output
algorithm: str = args.algorithm
clusters: int = args.clusters
compactness: float = args.compactness
iterations: int = args.iterations
connected_components: bool = args.connected_components
