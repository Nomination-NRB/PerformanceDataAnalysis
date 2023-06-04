import json
import re
import heapq
import pandas as pd
import numpy as np
import time


def getKeyNum(df):
    # 提取 results 列中带有 "#" 号的字段
    pattern = r'#\w+'  
    results = df['results'].str.cat(sep=' ')  # 将所有 results 列的数据合并为一个字符串
    hashtags = set(re.findall(pattern, results))  # 使用正则表达式提取带 "#" 号的字段，并去重

    # 统计每个带 "#" 号的字段在整个文件中出现的次数
    hashtags_dict = {}
    for hashtag in hashtags:
        count = results.count(hashtag)
        hashtags_dict[hashtag] = count
    return hashtags_dict


def getTop10Key(hashtags_dict):
    # 获取字典中数量前十的字段
    top_n = 10  # 自定义获取前几个字段
    top_n_fields = heapq.nlargest(top_n, hashtags_dict, key=hashtags_dict.get)
    
    print("数量前十的字段：")
    keyList = []
    for field in top_n_fields:
        print("字段名: {:<30s} 出现次数: {:d}".format(field, hashtags_dict[field]))
        keyList.append(field)
    return keyList


def get_input_output_Speed_Multi(Tdf, fields, keys_to_extract):
    # 选择需要提取的字段
    df = Tdf.copy()
    
    # 对dimension列进行预处理
    df['dimension'] = df['dimension'].apply(lambda x: json.loads(x))
    
    for key in keys_to_extract:
        df[key] = df['dimension'].apply(lambda x: 'None' if x.get(key) in [None, ''] else x.get(key))

    # 对cvm_cpu进行数值化处理
    df['cvm_cpu'] = df['cvm_cpu'].apply(pd.to_numeric, errors='coerce').fillna(-1)
    # 对cvm_memory进行数值化处理，通过正则表达式提取出数字
    df['cvm_memory'] = df['cvm_memory'].apply(lambda x: re.findall(r'\d+', x)[0] if re.findall(r'\d+', x) else '').apply(pd.to_numeric, errors='coerce').fillna(-1)

    # 筛选出符合要求的行
    df_filtered = df[df['results'].apply(lambda x: any(field in x for field in fields))]  # 检查任一字段是否出现在 results 中

    # 从results中提取出field对应的值
    df_output = pd.DataFrame(df_filtered['results'].apply(lambda x: json.loads(x)).tolist(), index=df_filtered.index)
    df_output = df_output[fields]  # 提取多个字段的值
    df_output = df_output.fillna(0)  # 将缺失值填充为 0

    # 将input和output分别转成dataframe
    df_input = df_filtered[keys_to_extract + ['results_key']]
    df_output = pd.DataFrame(df_output)

    # 将input和output合并成一个dataframe
    df_result = df_input.join(df_output)
    df_result = df_result.reset_index(drop=True)

    return df_result


def replace_zero_with_stat(df_result, fields, stat='None'):
    if stat == 'mean':
        stat_values = df_result[fields].mean()
        stat_values = stat_values.apply(lambda x: round(x, 6))
        df_result[fields] = df_result[fields].replace(0, stat_values)
        return df_result
    elif stat == 'median':
        stat_values = df_result[fields].median()
        stat_values = stat_values.apply(lambda x: round(x, 6))
        df_result[fields] = df_result[fields].replace(0, stat_values)
        return df_result
    else:
        return df_result
    



import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MyDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.input[idx], dtype=torch.float32)  
        y = torch.tensor(self.output[idx], dtype=torch.float32)  
        return x, y


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(64, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.relu6 = nn.ReLU()
        
        self.fc7 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(0.5)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.xavier_uniform_(self.fc7.weight)
        
    def forward(self, x):
        def apply_layers(x, fc, bn, relu):
            x = fc(x)
            x = bn(x)
            x = relu(x)
            x = self.dropout(x)
            return x
        x = apply_layers(x, self.fc1, self.bn1, self.relu1)
        x = apply_layers(x, self.fc2, self.bn2, self.relu2)
        x = apply_layers(x, self.fc3, self.bn3, self.relu3)
        x = apply_layers(x, self.fc4, self.bn4, self.relu4)
        x = apply_layers(x, self.fc5, self.bn5, self.relu5)
        x = apply_layers(x, self.fc6, self.bn6, self.relu6)
        x = self.fc7(x)
        return x


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def modify_values(df, your_keys, your_operators, values):
    for key, operator, value in zip(your_keys, your_operators, values):
        df[key] = df[key].apply(lambda x: eval(f'x {operator} {value}'))
    return df


def parse_string(input_string):
    yourKey = []
    yourOperator = []
    value = []
    expressions = input_string.split(',')

    for expression in expressions:
        expression = expression.strip()
        # 使用正则表达式匹配键、操作符和值
        matches = re.findall(r'(#\w+)\s*([*/])\s*(\d+)', expression)
        for match in matches:
            yourKey.append(match[0])
            yourOperator.append(match[1])
            value.append(float(match[2]))

    return yourKey, yourOperator, value



FileName = 'filterFile.csv'
FilePath='data/{}'.format(FileName)                                            # 换数据集请修改此处

FileData=pd.read_csv(FilePath)
FileData=FileData[:200000]
keyNumDict=getKeyNum(FileData)
your_fields = getTop10Key(keyNumDict)

print(your_fields)

valid_fields = [field for field in your_fields if field in keyNumDict]
invalid_fields = set(your_fields) - set(valid_fields)
if invalid_fields:
    print("Fields {} not in dictionary, have been removed".format(invalid_fields))
your_fields = valid_fields


keys_to_extract = ['cvm_cpu', 'cvm_memory', 'cvm_cpu_qos', 'cvm_os_type', 'cvm_cpu_type', 'cvm_version','cvm_gpu_type','host_cpu_type','host_memory_type','tool_version','component_version','host_manufacturer_name','host_type']
df_result = get_input_output_Speed_Multi(FileData, your_fields, keys_to_extract)

# df_result = replace_zero_with_stat(df_result, your_fields)



tempKey = keys_to_extract.copy()
tempKey.append('results_key')
print(keys_to_extract)
print(tempKey)

inPutDF = df_result[tempKey]
outPutDF = df_result[your_fields]

print(inPutDF.shape)
print(outPutDF.shape)

inPutDF.loc[:, 'cvm_cpu'] = pd.to_numeric(inPutDF['cvm_cpu'])
inPutDF.loc[:, 'cvm_memory'] = pd.to_numeric(inPutDF['cvm_memory'])
one_hot_df = pd.get_dummies(inPutDF, columns=tempKey[2:]).loc[:, :]
X = one_hot_df.values
y = outPutDF.values
print(X.shape)
print(y.shape)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

input_size = X.shape[1]
output_size = y.shape[1]
print('input_size: ', input_size)
print('output_size: ', output_size)



# 标准化输入数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

model = MyModel(input_size, output_size).to(device)
# dataset = MyDataset(X_scaled, y_scaled)
dataset = MyDataset(X, y)


train_size = int(0.75 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 1000
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 25
early_stop_counter = 0

# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) scheduler.step()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10) scheduler.step(val_loss) 若使用请添加到val_losses.append(val_loss)下一行
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)


startTime = time.time()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    train_losses.append(train_loss)
    val_loss = evaluate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

endTime = time.time()
print('Training time: ', time.strftime("%H:%M:%S", time.gmtime(endTime - startTime)))
print('\n')

with open('loss/{}_loss.txt'.format(FileName), 'w') as f:
    for i in range(len(train_losses)):
        f.write(str(train_losses[i]) + ' ' + str(val_losses[i]) + '\n')



test_loss = evaluate(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")


torch.save(model.state_dict(), 'model/{}_model.pth'.format(FileName))


model = MyModel(input_size, output_size).to(device)
model.load_state_dict(torch.load('model/{}_model.pth'.format(FileName)))


model.eval()
predictions = []
targets = []
# L1损失函数
# criterion = torch.nn.L1Loss()




with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # outputs = Myinverse_transform(outputs, scaler)
        predictions.extend(outputs.tolist())
        # labels = Myinverse_transform(labels, scaler)
        targets.extend(labels.tolist())


df_predictions = pd.DataFrame({'Prediction': predictions, 'Target': targets})

for field in your_fields:
    target_field = field + '_Target'
    prediction_field = field + '_Prediction'
    index = your_fields.index(field)

    df_predictions[target_field] = df_predictions['Target'].apply(lambda x: round(x[index], 6))
    df_predictions[prediction_field] = df_predictions['Prediction'].apply(lambda x: round(x[index], 6))

df_predictions.drop(['Prediction', 'Target'], axis=1, inplace=True)
df_test = df_predictions.copy()


for field in your_fields:
    df_test[field + '_absScore'] = (1-(abs(df_test[field + '_Target'] - df_test[field + '_Prediction']) / df_test[field + '_Target']))*100


df_test.to_csv('TestResult/{}_test_result.csv'.format(FileName), index=False)


field_lengths = {field: len(df_test[df_test[field + '_Target'] != 0]) for field in your_fields}
print("Field lengths:", field_lengths)

KeyScoreDict = {field: len(df_test[(df_test[field + '_absScore'] >= 90) & (df_test[field + '_absScore'] <= 100)]) for field in your_fields}
print("KeyScore Dict:", KeyScoreDict)