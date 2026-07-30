"""Compatibility launcher for the current Gradio lighting demo.

The maintained implementation lives in module 6. Keeping this small wrapper
means ``python demo.py`` and ``start_demo.command`` both open the same frontend.
"""

from modules.module_06_demo_evaluation.src.app import main


if __name__ == "__main__":
    main()
