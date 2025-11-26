import mesa
import inspect

def find_class(module, name):
    for key, value in inspect.getmembers(module):
        if key == name:
            print(f"Found {name} in {module.__name__}")
            return
        if inspect.ismodule(value) and value.__name__.startswith('mesa'):
            find_class(value, name)

print("Searching for RandomActivation...")
# Try to import submodules that might contain it
try:
    from mesa import time
    print("Imported mesa.time")
except ImportError:
    print("Could not import mesa.time")

# recursive search is hard with just dir, let's try common locations
try:
    from mesa.time import RandomActivation
    print("Found in mesa.time")
except ImportError:
    print("Not in mesa.time")

try:
    from mesa.scheduler import RandomActivation
    print("Found in mesa.scheduler")
except ImportError:
    print("Not in mesa.scheduler")
