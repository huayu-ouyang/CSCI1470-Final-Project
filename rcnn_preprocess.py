import pandas as pd

# put data into format defined here: https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/
data = pd.DataFrame()

i = 0
for _,row in train.iterrows():
    width = row.width
    height = row.height
    cell_type = "None"
    if row.target == 1:
        cell_type = "Pneumonia"
    data['format'][i] = 'train_images/' + row.patientId + ',' + str(row.x) + ','
    + str(row.y) + ',' + str(row.x + width) + ',' + str(row.y + height) + ','
    + cell_type
    i += 1

data.to_csv('annotate.txt', header=None, index=None, sep=' ')
