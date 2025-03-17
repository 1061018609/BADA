import torch
import umap
import umap.plot
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# cityscapes_class_center = torch.load('/home/jerry/xxz/DANNet/prototypes_cityscapes_44.6mIoU')
# darkzurich_night_class_center = torch.load('/home/jerry/xxz/DANNet/prototypes_darkzurich_night_44.6mIoU')
# darkzurich_day_class_center = torch.load('/home/jerry/xxz/DANNet/prototypes_darkzurich_day_44.6mIoU')

vectors_memory_name = 'vectors_memory_PSPNet_cityscapes_test_150000_cityscapes'

ACDC_class_center = torch.load('prototype/prototypes_nopairactive_19000')
# ACDC_class_vectors_list = torch.load('vectors_memory_nopairactive_19000')
ACDC_class_vectors_list = torch.load('prototype/{}'.format(vectors_memory_name))
for i in range(len(ACDC_class_vectors_list)):
    if i == 0:
        ACDC_class_features_tensor = ACDC_class_vectors_list[i]
        label = torch.Tensor([i]*ACDC_class_vectors_list[i].size()[0]).unsqueeze(0)
    else:
        ACDC_class_features_tensor = torch.cat([ACDC_class_features_tensor, ACDC_class_vectors_list[i]], 0)
        label = torch.cat([label, torch.Tensor([i]*ACDC_class_vectors_list[i].size()[0]).unsqueeze(0)], 1)
print(ACDC_class_center.size())
print(ACDC_class_features_tensor.size())

class_center_dict = {'road': 1, 'sidewalk': 1, 'building': 1, 'wall': 1, 'fence': 1, 'pole': 1, 'traffic light': 1, 'traffic sign': 1, 
               'vegetation': 1, 'terrain': 1, 'sky': 1, 'person': 1, 'rider': 1, 'car': 1, 'truck': 1, 'bus': 1, 'train': 1, 'motorcycle': 1, 'bicycle': 1}
class_name_list = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# label = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).resize(19,1)
label = label.resize(ACDC_class_features_tensor.size()[0], 1)
label = label.squeeze(1)
print(label.size())

# cos_sim_city_zurichday = torch.cosine_similarity(cityscapes_class_center, darkzurich_day_class_center, dim=1)
# for i in range(19):
#     class_center_dict[class_name_list[i]] = cos_sim_city_zurichday[i]#.numpy()
# print(class_center_dict)

# cos_sim_city_zurichnight = torch.cosine_similarity(cityscapes_class_center, darkzurich_night_class_center, dim=1)
# for i in range(19):
#     class_center_dict[class_name_list[i]] = cos_sim_city_zurichnight[i]#.numpy()
# print(class_center_dict)

# cos_sim_zurichday_zurichnight = torch.cosine_similarity(darkzurich_day_class_center, darkzurich_night_class_center, dim=1)
# for i in range(19):
#     class_center_dict[class_name_list[i]] = cos_sim_zurichday_zurichnight[i]#.numpy()
# print(class_center_dict)


mapper = umap.UMAP().fit(ACDC_class_features_tensor.numpy())
umap.plot.points(mapper, labels = label.numpy())
# plt.show()
plt.savefig('{}.jpg'.format(vectors_memory_name))






# digits = load_digits()
# print(digits.data.shape)
# print(digits.target.shape)
# exit()
# mapper = umap.UMAP().fit(digits.data)
# umap.plot.points(mapper, labels = digits.target)
