import json
surgeries = {"Foot Surgery": ["pincer", "shovel"]}

with open("surgical_data.json", "w") as write_file:
    json.dump(surgeries, write_file, indent=4)
