"""Command-line SRL interface.

Example usage:
```
python -m srl --text "Der Hund jagt die Katze."
```
"""
from .predict import main

if __name__ == "__main__":
    main()
