import os
import sys

def main() -> None:
    key = os.getenv("API_KEY", "").strip()
    if not key:
        print("API key not found. Set secret 'API_KEY' in GitHub Actions.", file=sys.stderr)
        sys.exit(1)

    # Do NOT print the key
    print("API key accessed successfully.")

if __name__ == "__main__":
    main()
