"""A standard-library CLI example; no model, service or file writes."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--allow-record", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    print(json.dumps({"steps": args.steps, "allow_record": args.allow_record}))


if __name__ == "__main__":
    main()
