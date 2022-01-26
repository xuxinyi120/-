import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='mine', split='train', clip_len=16, preprocess=False):
        #初始化类VideoDataset,并设置一些参数和参数默认值
        self.root_dir, self.output_dir = Path.db_dir(dataset)#获取数据集的源路径和输出路径
        folder = os.path.join(self.output_dir, split)# 获取对应分组的的路径
        self.clip_len = clip_len# 16帧图片的意思
        self.split = split# 有三组 train val test

        # The following three parameters are chosen as described in the paper section 4.1
        # 图片的高和宽的变化过程（h*w-->128*171-->112*112）
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        #生成视频对应的帧视频数据集
        # check_integrity()判断是否存在Dataset的源路径，若不存在，则报错
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # check_preprocess()判断是否存在Dataset的输出路径，若不存在preprocess()则创建,并在其中生成对应的帧图片的数据集
        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # 生成视频动作标签的txt文档
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        # fnames-->所有类别里的动作视频的集合; labels-->动作视频对应的标签
        self.fnames, labels = [], []
        print(folder)
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        #print(labels)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

         # Prepare a mapping between the label names (strings) and indices (ints)--> label和对应的数字标签
        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        #print(self.label2index)
         # Convert the list of label names into an array of label indices-->转化为数字标签
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        #print(self.label_array)

        # 生成对应的动作和数字标签的txt文档
        if dataset == "mine":
            if not os.path.exists('/mnt/nvme1n1/xxy/LRCN/mydataset_labels.txt'):
                with open('/mnt/nvme1n1/xxy/LRCN/mydataset_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    # 返回所有动作视频的总数
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])#加载一个视频生成的帧图片[frames,h,w,3]-->[frames,128,171,3]
        buffer = self.crop(buffer, self.clip_len, self.crop_size)# [16,112,112,3]
        labels = np.array(self.label_array[index])#  转化为数组

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)# 增强数据集
        buffer = self.normalize(buffer) # 归一化
        buffer = self.to_tensor(buffer)# [3,16,112,112]
        return torch.from_numpy(buffer), torch.from_numpy(labels) #以数组的形式返回

    # check_integrity()判断是否存在Dataset的源路径，若不存在，则报错
    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    # 检查输出路径是否存在，若不存在，则报错；检查输出路径的数据集图片格式是否正确，若不正确则报错
    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        # 创建对应的分组路径
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            #os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets-->划分train/val/test的数据集 0.6/0.2/0.2
        # Split train/val/test sets
        for file in os.listdir(self.root_dir): #file应该就是t和f
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train, test = train_test_split(video_files, test_size=0.4, random_state=42)
            #train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            #val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            #if not os.path.exists(val_dir):
            #    os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)#把视频转化为数组的形式表示

            #for video in val:
            #    self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]# 获取是视频名
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename)) # 创建视频对应的文件夹

        #读视频
        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        # 读取视频的帧数、高和宽
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #确保视频至少16帧
        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        # 把视频的一帧的高和宽修改成128.171，并命名保存.jpg的图片
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        #释放资源
        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        # 数据集以0.5的概率翻转，增强数据集
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    # [0,1,2,3]-->[3,0,1,2]  进行维度的变换
    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    # #加载一个视频生成的帧图片[frames,h,w,3]-->[frames,128,171,3]
    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='mine', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break