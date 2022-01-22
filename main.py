from pathlib import Path
from argparse import ArgumentParser
from image_segment.segment import ImageSegment


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("path", metavar="image", type=str, help="Path to image file.")
    args = parser.parse_args()
    path = Path(args.path).resolve()
    if not path.is_file():
        raise ValueError(f"{path} is not a valid file.")
    ImageSegment(path)


if __name__ == "__main__":
    main()
