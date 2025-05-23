import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

with open('./output/2025_04_12_12_08_21/dev_loss_result.pkl', 'rb') as f:
    losses = pickle.load(f)

recalls = []

with open('./output/2025_04_12_12_08_21/validation_epochs.txt', 'r') as f:
    for line in f:
        temp = {}
        data = map(str.strip, line.rstrip().split('|'))

        for d in data:
            k, v = map(str.strip, d.split(':'))
            temp[k] = int(v) if k == 'epoch' else float(v)

        temp['rsum'] = sum([y for x, y in temp.items() if x != 'epoch'])
        recalls.append(temp)

losses.sort(key=lambda x: x['epoch'])
recalls.sort(key=lambda x: x['epoch'])

rsum_list = np.array([i['rsum'] for i in recalls])
loss_list = np.array([i['loss'] for i in losses])

rsum_list = np.expand_dims(rsum_list, 1)
loss_list = np.expand_dims(loss_list, 1)

scaler_rsum = MinMaxScaler()
scaler_loss = MinMaxScaler()

rsum_list = scaler_rsum.fit_transform(rsum_list)
loss_list = scaler_loss.fit_transform(loss_list)

best_epoch = rsum_list.argmax()

rsum_list = np.squeeze(rsum_list, 1)
loss_list = np.squeeze(loss_list, 1)

plt.plot(range(len(rsum_list)), rsum_list, color='tab:blue', label='rsum')
plt.plot(range(len(loss_list)), loss_list, color='tab:green', label='loss')
plt.axvline(best_epoch, color='tab:red', alpha=0.5)
plt.title('rsum & loss')
plt.legend()
plt.savefig('./plot/recall_loss_plot.png', dpi=300)