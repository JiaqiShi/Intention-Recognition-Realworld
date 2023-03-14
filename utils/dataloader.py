from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_intention_dataloader_day(data_path,
                                   features,
                                   labels,
                                   split_method='day',
                                   cut_frame=0,
                                   batch_size=2,
                                   **kw):

    train_set, val_set, test_set = [
        ATR_dataset_all_new(data_path,
                            features,
                            labels,
                            sub_set,
                            split_method=split_method,
                            cut_frame=cut_frame,
                            **kw) for sub_set in [0, 1, 2]
    ]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


class ATR_dataset_all_new(Dataset):
    def __init__(self,
                 data_path,
                 features,
                 labels,
                 sub_set=-1,
                 split_method='all',
                 cut_frame=0,
                 **kw):
        super().__init__()

        self.data_path = data_path
        self.features = features
        self.labels = labels
        self.feature_list = []
        if cut_frame is None:
            self.cut_frame = 0
        else:
            self.cut_frame = cut_frame

        for feature in features:
            self.feature_list.append(self._load_feature(feature))

        self.ids_all, self.label_list = self._load_label(labels)

        if sub_set != -1:
            if split_method == 'all':
                train_val_ids, test_ids = train_test_split(self.ids_all,
                                                           test_size=0.2,
                                                           random_state=0)
                train_ids, val_ids = train_test_split(train_val_ids,
                                                      test_size=0.125,
                                                      random_state=1)

                self.ids = [train_ids, val_ids, test_ids][sub_set]
            else:
                print('Split using different days.')
                days = list(dict.fromkeys([i[:4] for i in self.ids_all]))
                train_val_days, test_days = train_test_split(days,
                                                             test_size=0.2,
                                                             random_state=0)
                train_days, val_days = train_test_split(train_val_days,
                                                        test_size=0.125,
                                                        random_state=0)
                self.days = [train_days, val_days, test_days][sub_set]
                self.ids = [v for v in self.ids_all if v[:4] in self.days]
        else:
            self.ids = self.ids_all

        self._clear()

        if self.cut_frame is not None and self.cut_frame > 0:
            self._cut_frame()

        self._padding()

        self.len = len(self.ids)

        print(self.get_shape())

    def get_shape(self):
        one_x = self[0][0]
        shapes = []
        for one_inp in one_x:
            if type(one_inp) == list:
                shapes.append(one_inp[0].shape)
            else:
                shapes.append(one_inp.shape)
        return shapes

    def _clear(self):
        feature_list = []
        for feature_dic in self.feature_list:
            fea = {k: v for k, v in feature_dic.items() if k in self.ids}
            feature_list.append(fea)

        self.feature_list = feature_list

        label_list = []
        for label_dic in self.label_list:
            lab = {k: v for k, v in label_dic.items() if k in self.ids}
            label_list.append(lab)

        self.label_list = label_list

    def _cut_frame(self):

        for num, feature in enumerate(self.features):

            def cut_frame(dic, cut_frame):
                dic_ = {}
                for k, v in dic.items():
                    length = len(v)
                    if cut_frame < length - 1:
                        dic_[k] = v[:-cut_frame]
                    else:
                        dic_[k] = v[:2]
                return dic_

            if feature in ['point_2d', 'point_3d', 'point_3d_rotated']:
                self.feature_list[num] = cut_frame(self.feature_list[num],
                                                   self.cut_frame)

    def _padding(self):
        def pad_dic(dic, max_len):
            dic_ = {}
            for k, v in dic.items():
                if type(v) == list:
                    v = np.array(v)
                if len(v.shape) == 3:
                    dic_[k] = [
                        np.pad(v, ((0, max_len - len(v)), (0, 0), (0, 0))),
                        len(v)
                    ]
                elif len(v.shape) == 2:
                    dic_[k] = [
                        np.pad(v, ((0, max_len - len(v)), (0, 0))),
                        len(v)
                    ]
            return dic_

        for num, feature in enumerate(self.features):
            if feature in ['point_2d', 'point_3d', 'point_3d_rotated']:
                max_len = 651
                self.feature_list[num] = pad_dic(self.feature_list[num],
                                                 max_len)
            elif feature == 'yolo_all':
                max_len = 14
                self.feature_list[num] = pad_dic(self.feature_list[num],
                                                 max_len)
            elif feature == 'yolo_self':
                max_len = 8
                self.feature_list[num] = pad_dic(self.feature_list[num],
                                                 max_len)

    def __len__(self):
        return self.len

    def _load_feature(self, feature):

        if feature in ['centroid_feature', 'last_res_all', 'last_res_self']:
            f_path = os.path.join(
                self.data_path,
                feature + '_cut' + str(self.cut_frame) + '_dir.pkl')
        elif feature == 'last_yolo_all':
            with open(os.path.join(self.data_path,
                                   'yolo_features_path.txt')) as f:
                f_path = f.read().split()[0]
        elif feature == 'last_yolo_self':
            with open(os.path.join(self.data_path,
                                   'yolo_features_path.txt')) as f:
                f_path = f.read().split()[1]
        else:
            f_path = os.path.join(self.data_path, feature + '_dir.pkl')

        with open(f_path, 'rb') as f:
            feature_dir = pickle.load(f)

        return feature_dir

    def _load_label(self, labels):
        result_csv = pd.read_csv(os.path.join(self.data_path,
                                              'labels_all.csv'),
                                 index_col=0)
        result_csv = result_csv[result_csv['invalid'] == -1]
        ids = result_csv.index.to_list()
        label_list = []
        for label in labels:
            if label == 'tem':
                tem = result_csv['temperature (0:no,1:yes)'].to_numpy().astype(
                    int).tolist()
                tem = dict(zip(ids, tem))
                label_list.append(tem)
            elif label == 'san':
                san = result_csv['sanitize (0:no,1:yes)'].to_numpy().astype(
                    int).tolist()
                san = dict(zip(ids, san))
                label_list.append(san)
            elif label == 'tem_or_san':
                tem_or_san = np.logical_or(
                    result_csv['temperature (0:no,1:yes)'].to_numpy(),
                    result_csv['sanitize (0:no,1:yes)'].to_numpy()).astype(
                        int).tolist()
                tem_or_san = dict(zip(ids, tem_or_san))
                label_list.append(tem_or_san)
        assert len(label_list) > 0
        return ids, label_list

    def __getitem__(self, index):
        id = self.ids[index]
        return [dic[id] for dic in self.feature_list
                ], [dic[id] for dic in self.label_list], id
