import json
#@souce- https://www.geeksforgeeks.org/json-load-in-python/
#@source- https://realpython.com/lessons/serializing-json-data/
#opening json file
f = open("surgical_data.json",)

#returns json object as dictionary
surgeries=json.load(f)

#closing file
f.close()
surgeries["Eye Surgery"]= {"pincer": 1, "scissor_length1": 2, "scissor_length2": 1}
with open("surgical_data.json", "w") as write_file:
    json.dump(surgeries, write_file, indent=4)
