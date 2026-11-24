"""Child entry point for one study cell, kept out of the package import graph.

``xrl.studies.__init__`` imports :mod:`xrl.studies.cell_worker`, so running that
module with ``-m`` would load it once as a package submodule and again as
``__main__``. Runpy warns about exactly that, and every worker log would open
with the warning. This shim is imported by nothing, so ``-m`` executes it once.
"""

from __future__ import annotations

from xrl.studies.cell_worker import main

if __name__ == "__main__":
    raise SystemExit(main())
