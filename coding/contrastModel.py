import re
import os
import pandas as pd

def contrastModel(outputFilePath = 'output/'):
    fileList = os.listdir(outputFilePath)
    fileNumber = len(fileList)
    fileName= []
    for i in range(fileNumber):
        fileName.append(fileList[i].split('.')[0])
        
    # 存储提取的值
    result = []
    for file in fileList:
        filePath = os.path.join(outputFilePath, file)
        with open(filePath, 'r', encoding='utf-8') as f:
            content = f.read()
            field_lengths_matches = re.findall(r"Field lengths: \{([^}]+)\}", content)
            key_score_dict_matches = re.findall(r"KeyScore Dict: \{([^}]+)\}", content)
            result.append([field_lengths_matches, key_score_dict_matches])


    keyNameList = []
    tempResult = result[0][0]
    tempNameSingle = [item.strip() for sublist in [item.split(',') for item in tempResult] for item in sublist]
    for i in range(len(tempNameSingle)):
        keyNameList.append(tempNameSingle[i].split(':')[0])

  
    # columnName的值由keyNameList[i] + '_Number' 和 keyNameList[i] + '_Score'组成
    indexName = []
    for i in range(len(keyNameList)):
        indexName.append(keyNameList[i] + '_Number')
        indexName.append(keyNameList[i] + '_Score')


    valueList = []
    for i in range(len(result)):
        tempList = []
        for j in range(len(result[i][0])):
            number_match = re.findall(r": \d+", result[i][0][j])
            score_match = re.findall(r": \d+", result[i][1][j])
            numberList = []
            scoreList = []
            for k in range(len(number_match)):
                numberList.append(int(number_match[k].split(': ')[1]))
                scoreList.append(int(score_match[k].split(': ')[1]))
            for number in numberList:
                tempList.append(number)
            for score in scoreList:
                tempList.append(score)
        valueList.append(tempList)


    df = pd.DataFrame(columns=fileName, index=indexName)
    for i in range(len(fileName)):
        df.loc[:,fileName[i]] = valueList[i]
    
    return df


df = contrastModel()
df.head(20)