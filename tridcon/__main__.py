#!/usr/bin/env python3
#
if __package__ is None or __package__ == '':
    from cli import main
else:
    from .cli import main

main()
