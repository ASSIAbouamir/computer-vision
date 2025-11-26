import mesa
print(dir(mesa))
try:
    import mesa.time
    print("mesa.time exists")
except ImportError:
    print("mesa.time does not exist")
