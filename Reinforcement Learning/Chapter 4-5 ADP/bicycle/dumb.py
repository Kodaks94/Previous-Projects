import csv
with open("insurance.csv") as insuranceFile:
    insuranceData = csv.DictReader(insuranceFile)
    insuranceDict = {}
    for row in insuranceData:
        for column, value in row.items():
            insuranceDict.setdefault(column,[]).append(value)


regions = ["southwest","southeast","northeast","northwest"]
new_dict = {"southwest":{},"southeast":{},"northeast":{},"northwest":{}}
for i in regions:
    indices  = [b for b, x in enumerate(insuranceDict["region"]) if x == i]
    new_dict[i] = {k:[v[b] for b in indices ] for k,v in insuranceDict.items()}
