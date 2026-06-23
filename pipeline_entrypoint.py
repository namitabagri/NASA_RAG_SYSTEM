"""Legacy entrypoint preserving backwards compatibility.

This module delegates execution to the new CLI implementation while keeping
`pipeline_entrypoint.py` usable as a script entrypoint.
"""

from pipeline_cli import main

if __name__ == '__main__':
    main()
