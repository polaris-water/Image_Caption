# -*- coding: UTF-8 -*-  
# @Time : 2021/12/10 21:04
# @Author : GCR
# @File : 01_对人工描述进行预处理.py
# @Software : PyCharm
# coding:utf8
import torch as t
import numpy as np
import json
import jieba
import tqdm


# 配置类。 设置一些配置，其中每个配置都是类的属性，可通过对象的方式进行调用
class Config:
    annotation_file = r'data/caption_validation_annotations_20170910.json'  # 图片注释文件的位置
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    max_words = 5000
    min_appear = 2
    save_path = r'caption_2.pth'


# START='</START>'
# MAX_LENS = 25,

def process(**kwargs):
    opt = Config()  # 配置对象
    for k, v in kwargs.items():
        setattr(opt, k, v)    # 设置属性

    with open(opt.annotation_file) as f:
        data = json.load(f)         # 将图片注释文件引入程序中，看成对象，对象名为 data

    # 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg -> 0
    id2ix = {item['image_id']: ix for ix, item in enumerate(data)} # 得到一个字典，键：image_id的值，值：给图片编号（0--29999）
    # 0-> 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg
    ix2id = {ix: id for id, ix in (id2ix.items())} # 得到一个字典，键：给图片的编号（0-29999）   值：image_id的值
    assert id2ix[ix2id[10]] == 10  # 断言没问题，程序才能继续执行

    # 得到二维列表（30000x5），列表的每一行表示一张图片的描述，行的每个元素表示一句话。
    captions = [item['caption'] for item in data]

    #得到一个三维列表（30000x5xMax_len）。其中每个二维表是一张图片分词，二维表每一行表示一句描述的分词。
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]

    word_nums = {}  # 一个一维的字典，记录每个词出现的数量

    #返回一个 fun 函数，用于更新 word_num 中每个词出现的数量。fun用于接收一个词，然后更新 word_nums中这个词出现的次数
    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None

        return fun

    lambda_ = update(word_nums) # lambda_ 相当于update中的 fun 函数
    # 将 cut_captions 中的分的所有词传入 fun() 中，更新word_nums
    _ = {lambda_(word) for sentences in cut_captions for sentence in sentences for word in sentence}

    # [ (10000,u'快乐')，(9999,u'开心') ...]：得到一个二维元组列表，列表的每一行表示一个元组
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)

    #### 以上的操作是无损，可逆的操作###############################
    # **********以下会删除一些信息******************

    # 1. 丢弃词频不够的词
    # 2. ~~丢弃长度过长的词~~

    #返回一个列表，列表中保存着出现次数从多到少，且每个词至少出现两次的 前 5000个词。
    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)} # 为每个词添加索引（索引号越小，词出现次数越多），返回一个字典，键：词  值：词的索引
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    # 还是三维列表。将分词结果中的每个词，用上面得到的词的序号来表示
    ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence]
                    for sentence in item]
                   for item in cut_captions]
    readme = u"""
    word：词
    ix:index
    id:图片名
    caption: 分词之后的描述，通过ix2word可以获得原始中文词
    """

    #resutls：存储各种对象的字典。键：对象名  值：对象
    results = {
        'caption': ix_captions, # 三维列表。 三万张图片的描述的分词结果，每个词用序号表示
        'word2ix': word2ix,  # 一维字典。每个词对应的序号（序号越小，词出现的次数越多）
        'ix2word': ix2word,  # 一维字典。每个序号对应的词（序号越小，词出现的次数越多）
        'ix2id': ix2id,     # 一维字典。 每个序号对应的 图片的 image_id （序号是图片的存储顺序）
        'id2ix': id2ix,    # 一维字典。每张图片的 iamge_id（就是每张图片）  对应的序号
        'padding': '</PAD>', # pad标识符
        'end': '</EOS>',   # 结束标识符
        'readme': readme
    }

    t.save(results, opt.save_path) # 将结果保存为 pth 文件
    print('save file in %s' % opt.save_path)

    # 测试是否成功
    def test(ix, ix2=4):
        results = t.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][4] # 第1000张图片的第四句描述（分词状态，每个词用序号表示）
        sentences_p = (''.join([ix2word[ii] for ii in examples])) # 转成汉字
        sentences_r = data[ix]['caption'][ix2] # 从原始数据 data 中提取出第四句描述
        assert sentences_p == sentences_r, 'test failed'

    test(1000) # 测试对象：第1000张图片
    print('test success')


if __name__ == '__main__':
    process()
