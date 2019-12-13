
import numpy as np
import os
from glob import glob
import csv
from torch.utils.data import Dataset
from PIL import Image
import torch as t
import torchvision.transforms as T
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable as V
from torch.optim import lr_scheduler
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from incepForOCN import inception_v3

base_bone_dir = os.path.join('..', 'input' ,'rsna-bone-age')
dataPath = os.path.join(base_bone_dir, 'boneage-training-dataset', 'boneage-training-dataset')
labelPath = os.path.join(base_bone_dir, 'boneage-training-dataset.csv')

label = csv.reader(open(labelPath, 'r'))

male_data = [[name, age] for (name, age, male) in label if male == "True"]

for item in male_data:
    item[0] = os.path.join(dataPath, item[0] + ".png")



imgs_num = len(male_data)
# random.shuffle(male_data)
# np.random.seed(100)
# imgs = np.random.permutation(imgMale)

train_male = male_data[:int(0.8 * imgs_num)]
val_male = male_data[int(0.8 * imgs_num):int(0.9 * imgs_num)]
test_male = male_data[int(0.9 * imgs_num):]

# print(" train size: ", len(train_male))
# print(" val size: ", len(val_male))
# print(" test size: ", len(test_male))


class dataset(Dataset):
    def __init__(self, data_label, transforms):
        self.data_label = data_label
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.data_label[index][0]
        age_label = self.data_label[index][1]

        # label processing
        # In this code implementation,label number indexs of subsequences is start from 0 
		# In paper the index is start from 1
		# This is for easier code implementation. 
		# There is no conflict and essential difference between this and the principles in the paper
        age_left = (np.asarray((int(age_label) + 1) // 3))
        age_mid = (np.asarray((int(age_label) // 3)))
        age_right = (np.asarray((int(age_label) - 1) // 3))

        age_left = t.from_numpy(age_left)
        age_mid = t.from_numpy(age_mid)
        age_right = t.from_numpy(age_right)

        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)

        return img_path, data, age_left, age_mid, age_right

    def __len__(self):
        return len(self.data_label)





USE_GPU = t.cuda.is_available()

transform1 = T.Compose([
    T.RandomRotation(15),
    T.Resize((299, 299)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    T.ToTensor(),
    T.Normalize([0.5, ], [0.5, ])
])

transform2 = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize([0.5, ], [0.5, ])
])

train_dataset = dataset(data_label=train_male, transforms=transform1)
val_dataset = dataset(data_label=val_male, transforms=transform2)
test_dataset = dataset(data_label=test_male, transforms=transform2)

path, data, age1, age2, age3 = train_dataset[0]

print("path:", path)
print(type(data))
print(age1)
print(age2)
print(age3)
all_dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

data_loader = {x: t.utils.data.DataLoader(all_dataset[x], batch_size=16, shuffle=True) for x in
               ['train', 'val', 'test']}

model = inception_v3(pretrained=False, aux_logits=False)
model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.005)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 14, gamma = 0.33)

EPOCH = 80
BATCH_SIZE = 16
best_month = 228
erroloss_weight = 0

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

for epoch in range(EPOCH):
    scheduler.step()
    print('Epoch {}/{}, lr = {}'.format(epoch, EPOCH - 1, optimizer.state_dict()['param_groups'][0]['lr']))
    print('-' * 15)
    for phase in ['train', 'val', 'test']:

        if phase == 'train':
            print('Training...')
            model.train(True)
        elif phase == 'val':
            print('Validating...')
            model.train(False)
        else:
            print('Test....')
            model.train(False)

        running_loss = 0.0
        month_mean = 0.0
        half_year_accuracy = 0.0
        one_year_accuracy = 0.0
        two_year_accuracy = 0.0
        left_oneyear = 0.0
        mid_oneyear = 0.0
        right_oneyear = 0.0
        for step, data in enumerate(data_loader[phase], 1):

            # print(step)
            _, data, age_left, age_mid, age_right = data

            if USE_GPU:
                data = V(data.cuda())
                age_left = V(age_left.cuda())
                age_mid = V(age_mid.cuda())
                age_right = V(age_right.cuda())
                model.cuda()
            # clear the grad

            optimizer.zero_grad()
            # forward
            output = model(data)
            #             print(output)
            #             print(output[:,:3])
            output_left = output[:, :77]
            output_mid = output[:, 77:154]
            output_right = output[:, 154:]
            pred_left = output_left.cpu().data.max(1, keepdim=True)[1].cuda()
            pred_mid = output_mid.cpu().data.max(1, keepdim=True)[1].cuda()
            pred_right = output_right.cpu().data.max(1, keepdim=True)[1].cuda()
            # Error Loss
            erroloss = (sum(abs(pred_left - pred_mid) // 2) + sum(abs(pred_mid - pred_right) // 2)) * erroloss_weight
            # Ovelap loss = Overlap classification loss + erroloss
            loss = criterion(output_left, age_left) + criterion(output_mid, age_mid) + criterion(output_right,
                                                                                                 age_right) + erroloss


            pred_left = pred_left.cpu().float().numpy()
            pred_mid = pred_mid.cpu().float().numpy()
            pred_right = pred_right.cpu().float().numpy()
			# In paper, which index start from 1 , preid_age = pred_left + pred_age + pred_mid - 2
			# In this code implementation , which label index start from 0, pred_age = pred_left + pred_right + pred_mid + 1
            pred_age = pred_left + pred_right + pred_mid + 1



            age_left = age_left.cpu().view(age_left.shape[0], 1).numpy()
            age_mid = age_mid.cpu().view(age_left.shape[0], 1).numpy()
            age_right = age_right.cpu().view(age_right.shape[0], 1).numpy()
            age_label = age_left + age_right + age_mid + 1


            month_mean += float(sum(abs(pred_age - age_label)))
            #             half_year_accuracy += int(sum(abs(target_age == pred_age)))
            half_year_accuracy += int(sum(abs(pred_age - age_label) <= 6))
            one_year_accuracy += int(sum(abs(pred_age - age_label) <= 12))
            two_year_accuracy += int(sum(abs(pred_age - age_label) <= 24))

            left_oneyear += int(sum(abs(pred_left - age_left) <= 4))
            mid_oneyear += int(sum(abs(pred_mid - age_mid) <= 4))
            right_oneyear += int(sum(abs(pred_right - age_right) <= 4))

            # backward
            if phase == 'train':
                loss.backward()
                optimizer.step()
                # accumulate the loss and correct amounts
            running_loss += loss.item()

            if step % 100 == 0 and phase == 'train':
                print(
                    "Train Epoch:{}  print_freq:{} Loss:{:.4f}  month_mean: {:4f},  half_year: {:4f},"
                    " one_year: {:4f}, two: {:4f}, left_one: {:4f}, right_one: {:4f}".format(epoch,
                                                                                             step,
                                                                                             running_loss / ((
                                                                                                                     step + 1) * BATCH_SIZE),
                                                                                             month_mean / ((
                                                                                                                   step + 1) * BATCH_SIZE),
                                                                                             half_year_accuracy / ((
                                                                                                                           step + 1) * BATCH_SIZE),
                                                                                             one_year_accuracy / ((
                                                                                                                          step + 1) * BATCH_SIZE),
                                                                                             two_year_accuracy / ((
                                                                                                                          step + 1) * BATCH_SIZE),
                                                                                             left_oneyear / ((
                                                                                                                     step + 1) * BATCH_SIZE),
                                                                                             right_oneyear / ((
                                                                                                                      step + 1) * BATCH_SIZE)))

        epoch_loss = float(running_loss) / float(len(data_loader[phase]) * BATCH_SIZE)
        month_mean = float(month_mean) / float(len(data_loader[phase]) * BATCH_SIZE)
        epoch_half_year_accuracy = float(half_year_accuracy) / float(len(data_loader[phase]) * BATCH_SIZE)
        epoch_one_year_accuracy = float(one_year_accuracy) / float(len(data_loader[phase]) * BATCH_SIZE)
        epoch_two_year_accuracy = float(two_year_accuracy) / float(len(data_loader[phase]) * BATCH_SIZE)
        epoch_left_one = float(left_oneyear) / float(len(data_loader[phase]) * BATCH_SIZE)
        epoch_right_one = float(right_oneyear) / float(len(data_loader[phase]) * BATCH_SIZE)

        f = open('./checkpoint/log.txt', 'a+')
        f.write("Epoch: {}, Phase: {}, Loss: {:.4f}, month_mean: {:4f}, half_year: {:4f},"
                "  one_year: {:4f},  two: {:4f}\n".format(epoch, phase,
                                                          epoch_loss,
                                                          month_mean,
                                                          epoch_half_year_accuracy,
                                                          epoch_one_year_accuracy,
                                                          epoch_two_year_accuracy,
                                                          ))
        f.close()
        print("Train Epoch :  {}    Loss: {:.4f}  ,  month_mean: {:4f},  half_year_accuracy: {:4f},"
              "  one_year_accuracy: {:4f},  two: {:4f}, left_one: {:4f}, right_one: {:4f} ".format(epoch,
                                                                                                   epoch_loss,
                                                                                                   month_mean,
                                                                                                   epoch_half_year_accuracy,
                                                                                                   epoch_one_year_accuracy,
                                                                                                   epoch_two_year_accuracy,
                                                                                                   epoch_left_one,
                                                                                                   epoch_right_one))

        if phase == 'val' and month_mean < best_month:
            best_month = month_mean
            f = open('./checkpoint/best_log.txt', 'a+')
            f.write(' Epoch: {}, month_mean: {}\n'.format(epoch, best_month))
            f.close()