# -*- coding: UTF-8 -*-  
# @Time : 2021/12/10 21:22
# @Author : GCR
# @File : 00_全部代码.py
# @Software : PyCharm
import torch
from torch.utils import data
import numpy as np
from tqdm import *
from torch.nn.utils.rnn import pack_padded_sequence
from beam_search import CaptionGenerator
from PIL import Image
import torchvision as tv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torch.utils.data.dataset import random_split
import dill as pickle


def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.fc(x)
    return x


model_feature = resnet50(pretrained=True)
model_feature.forward = lambda x: new_forward(model_feature, x)
model_feature = model_feature.cuda()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
transforms = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(256),
    tv.transforms.ToTensor(),
    normalize
])


# 重新定义数据类
class CaptionDataset(data.Dataset):
    def __init__(self):
        data = torch.load('caption_2.pth')
        # ix2word = data['ix2word']
        self.ix2id = data['ix2id']  # 每个序号对应的图片的iamge_id
        self.caption = data['caption']  # 三万张图片的描述的分词结果，每个词用序号表示
        word2ix = data['word2ix']  # 每个词对应的序号（序号越小，词出现的次数越多）
        self.padding = word2ix.get(data.get('padding'))  # 'padding'在word2ix字典中对应的序号为 1
        self.end = word2ix.get(data.get('end'))  # 'end'在word2ix字典中对应的序号为 2

        self.feature = torch.load('results_2048.pth')

    def __getitem__(self, item):
        img = self.feature[item]  # 该图片的特征表示
        caption = self.caption[item]  # 该图片的文本描述
        rdn_index = np.random.choice(len(caption), 1)[0]  # 5句描述随机选一句
        caption = caption[rdn_index]  # 随机选的描述
        return img, torch.LongTensor(caption), item  # 特征表示，某一句描述，图片序号

    def __len__(self):
        return len(self.ix2id)


#将 batch_size 个样本整理成一个 batch 样本
def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        """
        将batch_size个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]

        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>

        返回：
        - imgs(Tensor): batch_sie*2048
        - cap_tensor(Tensor): batch_size*max_length (I think it is wrong!)
        - lengths(list of int): 长度为 batch_size
        - index(list of int): 长度为batch_size
        """
        # print('hello world') # 这句没有执行-->好吧，看来函数没有执行。

        img_cap.sort(key=lambda p: len(p[1]), reverse=True) # 按照 各个图片的文本描述 的分词数量进行。降序。
        imgs, caps, indexs = zip(*img_cap)  # 训练的数据集。 imgs（二维张量）：图片的特征表示    cps（二维张量）：图片的某一句描述（分词表示）  indexs（一维张量）:图片的序号
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)  # batch * 2048 （二维张量）
        lengths = [min(len(c) + 1, max_length) for c in caps]  # 存储 图片描述 中 分词的数量
        batch_length = max(lengths)  # 分词数量的最大值（作为存储分词的张量的长度）

        #用于存储新的 图片的描述，刚开始，初始化为 padding
        #每一列，存储一句图片的 分词描述
        cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding) # 扩展存储分词的张量的长度为batch_length，空位置用padding表示

        #i：每一张图片的序号
        #c:这张图片的一句分词描述
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1  # 扩展前存储分词 的张量的最后一个位置。
            if end_cap < batch_length:  # 如果最后一个位置<存储分词的张量的长度
                cap_tensor[end_cap, i] = eos  # 在结尾添加 eos
            # print('c的形状为：')
            # print(c.shape)
            # print('cap_tensor的形状为：')
            # print(cap_tensor.shape)
            if min(c.shape)==0: # 出错的地方
                pass
            else:
                cap_tensor[:end_cap, i].copy_(c[:end_cap]) # 将原先cps中的内容复制到新的 captensor中去
            #如：若 i =0，将第一张图片的分词描述复制到 cap_tensor中第一列中从 0 到 end_cap 的位置。

        return (imgs, (cap_tensor, lengths), indexs)  # batch * 2048, (max_len * batch, ...), ...

    return collate_fn


batch_size = 32  # 一批数据32个
max_epoch = 50
embedding_dim = 64  # 每个词用 64维的向量表示
hidden_size = 64  #
lr = 1e-4
num_layers = 2  # 2 层的LSTM


def get_dataloader():
    '''
    dataset是一个元组的集合。一个元组表示一条数据，共30000条,30000个元组，每个元组内容为：
    元组中第一个元素：img：图片的特征表示（一个2048维的向量）（张量）
    元组中第二个元素：caption：图片的某一句描述（分词表示）  （张量）
    元组中第三个元素：item：图片的序号  （整型）
    '''
    dataset = CaptionDataset()
    n_train = int(len(dataset) * 0.9)  # 90% 作为训练数据（27000条）
    split_train, split_valid = random_split(dataset=dataset, lengths=[n_train, len(dataset) - n_train])  # 划分数据集
    train_dataloader = data.DataLoader(split_train, batch_size=batch_size, shuffle=True, num_workers=0   # 注意：num_workers=0，在我的电脑上这样才行
                                       , collate_fn=create_collate_fn(dataset.padding,
                                                                      dataset.end))  # collate_fn=create_collate_fn(dataset.padding, dataset.end)
    valid_dataloader = data.DataLoader(split_valid, batch_size=batch_size, shuffle=True, num_workers=0
                                       , collate_fn=create_collate_fn(dataset.padding,
                                                                      dataset.end))  # collate_fn=create_collate_fn(dataset.padding, dataset.end)
    return train_dataloader, valid_dataloader


class Net(torch.nn.Module):
    def __init__(self, word2ix, ix2word):
        super(Net, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.embedding = torch.nn.Embedding(len(word2ix), embedding_dim)  # 5002x64 的二维表
        self.fc = torch.nn.Linear(2048, hidden_size)  # 图像经过ResNet提取成2048维的向量，然后利用全连接层转成256维的向量
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers)  # 计算每个词的输出
        self.classifier = torch.nn.Linear(hidden_size, len(word2ix))  # 利用每个词的输出进行分类，预测下一个词（分类）

    def forward(self, img_feats, captions, lengths):
        embeddings = self.embedding(captions)  # seq_len * batch * embedding
        img_feats = self.fc(img_feats).unsqueeze(
            0)  # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样, 1 * batch * hidden_size
        embeddings = torch.cat([img_feats, embeddings],
                               0)  # 将img_feats看成第一个词的词向量, (1+seq_len) * batch * hidden_size，和其他词向量拼接在一起
        packed_embeddings = pack_padded_sequence(embeddings, lengths)  # PackedSequence, lengths - batch中每个seq的有效长度
        outputs, state = self.rnn(
            packed_embeddings)  # seq_len * batch * (1*256), (1*2) * batch * hidden_size, lstm的输出作为特征用来分类预测下一个词的序号, 因为输入是PackedSequence,所以输出的output也是PackedSequence, PackedSequence第一个元素是Variable,第二个元素是batch_sizes,即batch中每个样本的长度*
        pred = self.classifier(outputs[0])
        return pred, state


    # 生成图片的文本描述，主要是使用beam search算法以得到更好的描述
    def generate(self, img, eos_token='</EOS>', beam_size=3, max_caption_length=30,
                 length_normalization_factor=0.0):
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img = img.unsqueeze(0) #torch.Size[[2048]]
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx.item()] for idx in sent])
                     for sent in sentences]
        return sentences


def evaluate(dataloader): # dataloader 是验证集
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for ii, (imgs, (captions, lengths), indexes) in enumerate(dataloader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]
            score, _ = model(imgs, input_captions, lengths)
            loss = criterion(score, target_captions)
            total_loss += loss.item()
    model.train()
    return total_loss





if __name__ == '__main__':
    train_dataloader, valid_dataloader = get_dataloader()  # 得到训练集和验证集
    _data = torch.load('caption_2.pth')  # 对图片的文本描述
    word2ix, ix2word = _data['word2ix'], _data['ix2word']

    # max_loss = float('inf')     # 221
    max_loss = 263

    device = torch.device('cuda')

    model = Net(word2ix, ix2word)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

    losses = []        # 训练集的误差
    valid_losses = []  # 使用验证集在不同时刻下得到的误差总和

    # img_path = '123.jpg'
    # raw_img = Image.open(img_path).convert('RGB')
    # raw_img = transforms(raw_img)  # 3*256*256
    # img_feature = model_feature(raw_img.cuda().unsqueeze(0))
    # print(img_feature)



    for epoch in range(max_epoch):
        for ii, (imgs, (captions, lengths), indexes) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad() # 梯度置0，把 loss 关于weight的导数变为0
            imgs = imgs.to(device)  # torch.Size([32, 2048])
            captions = captions.to(device) # torch.Size([max_length, 32])
            input_captions = captions[:-1] # torch.Size([max_length-1,32])，因为最后一行要么是padding，要么是eos，没有分词数据。
            target_captions = pack_padded_sequence(captions, lengths)[0] # 消除padding标识符，合并为一维张量
            score, _ = model(imgs, input_captions, lengths)
            loss = criterion(score, target_captions)
            loss.backward() # 反向传播，计算当前梯度
            optimizer.step()# 根据梯度更新网络参数
            losses.append(loss.item())# 每次的误差加入到 losses中

            if (ii + 1) % 20 == 0:  # 可视化
                # 可视化原始图片 + 可视化人工的描述语句
                raw_img = _data['ix2id'][int(indexes[0].numpy())] # 当前batch中，第一个照片的序号对应的image_id
                img_path = r'F:\\09_周任务\\2021.12.5：让神经网络讲故事\\数据集\\ai_challenger_caption_validation_20170910\\caption_validation_images_20170910\\' + raw_img
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)
                #
                raw_caption = captions.data[:, 0] # 该图片的文本描述
                raw_caption = ''.join([_data['ix2word'][ii.item()] for ii in raw_caption])
                #
                results = model.generate(imgs.data[0]) # 传入该图片的特征表示，返回文本描述
                #
                print(img_path, raw_caption, results)
                #
                #
                # print(model.generate(img_feature.squeeze(0)))
                tmp = evaluate(valid_dataloader) # 使用验证集在当前模型下得到的误差总和
                valid_losses.append(tmp)
                if tmp < max_loss:
                    max_loss = tmp
                    torch.save(model.state_dict(),
                               'model_best.pth')
                    print(max_loss)  # 190 111

    plt.figure(1)
    plt.plot(losses)
    plt.figure(2)
    plt.plot(valid_losses)
    plt.show()

    # test()


    # model.load_state_dict(torch.load('/mnt/Data1/ysc/ai_challenger_caption_validation_20170910/model_best.pth'))
    # print(model.generate(img_feature.squeeze(0)))
