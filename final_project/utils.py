
def cleanAttributes(dataDict):
    for key in dataDict:
        for attr in dataDict[key]:
           if dataDict[key][attr] == 'NaN':
               dataDict[key][attr] = 0.0

def createRatioAttribute(dataDict, counterName, denominatorName, newName):
    for key in dataDict:
        if dataDict[key][denominatorName] > 0.0:
            dataDict[key][newName] = dataDict[key][counterName] / dataDict[key][denominatorName]
        else:
            dataDict[key][newName] = 0.0

    return newName