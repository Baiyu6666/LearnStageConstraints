import sys
from . import active, shell_exports

if len(sys.argv) > 1 and sys.argv[1] == "shell":
    print(shell_exports())
elif len(sys.argv) > 1 and sys.argv[1] == "hardware":
    active().require_hardware()
elif len(sys.argv) > 1 and sys.argv[1] == "network":
    active().check_network()
else:
    import json
    print(json.dumps(active().metadata(), indent=2))
