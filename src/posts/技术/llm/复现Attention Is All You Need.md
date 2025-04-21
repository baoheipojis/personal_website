---
icon: pen-to-square
date: 2022-01-09
category:
  - 技术

tag:
  - 未完工
  - AI
  - NLP
  - Transformer
---

# 复现Attention Is All You Need
## 前言
Attention Is All You Need这个论文相信懂点AI的人没有不知道的。它创新性的提出了Transformer架构，替代了之前的RNN和CNN架构，成为了NLP领域的主流模型，并且应用于CV等其它领域。

本文将介绍：
1. Encoder-Decoder架构
2. Attention机制

并提供一段可以运行的代码复现。

你需要拥有的前置知识：
1. 了解Python编程语言


### 为什么想要复现Attention Is All You Need
这不完全是出于自己的兴趣，是因为南大有一门“科研实践”课程，让大二和大三的同学们感受一下科研。导师是做NLP的，给了我们几个课题。包括复现Attention Is All You Need, ReAct等等，或者是跑一些线上研究，像是大模型推理能力、数学能力、常识等等。当时第一个就是复现Attention Is All You Need，看见她的第一眼，我就知道我不会再选别的课题了，而且我还很担心这个课题被别人抢走，所以老师问的时候我马上抢答了。

下面我们就开始复现了，在复现之前，先记住这个图：
![Transformer 结构](Transformer结构.jpg)
这是原论文里给出的图

## 模型结构
先到论文的最后部分检查一下超参数：

```Python
# model.py
import torch
import torch.nn as nn
import math
from encoder import EncoderLayer
from decoder import DecoderLayer
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Transformer 模型类，实现基于 Encoder-Decoder 架构的翻译模型。

    Args:
        d_mode (int): 模型隐藏层维度，也就是每个 token 的嵌入向量维度，通常设为 512。
        nhead (int): 多头注意力机制中的头数，必须能被 d_mode 整除，例如 512/8 = 64，每个头负责处理子空间的信息。
        num_encoder_layers (int): 编码器（Encoder）的层数，原论文中通常为 6。
        num_decoder_layers (int): 解码器（Decoder）的层数，原论文中通常为 6。
        d_ff (int): 前馈神经网络（Feed-Forward Network）的隐藏层维度，通常比 d_mode 大很多（例如 2048）。
        vocab_size (int): 词汇表大小，表示可处理的唯一 token 数量。根据具体任务和分词策略，一般为 32000 或其他数值。
        max_len (int): 模型能处理的最大序列长度，用于位置编码，通常设为 512。
    """
    def __init__(self, d_mode=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, vocab_size=32000, max_len=512):
        super(Transformer, self).__init__()
        
        self.input_embedding = nn.Embedding(src_vocab_size, d_model)  # 编码器端
        self.output_embedding = nn.Embedding(tgt_vocab_size, d_model) # 解码器端
        # 这是论文3.4节提到的缩放因子
        self.scale = math.sqrt(d_model)        # added scale factor
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 通过 Encoder 层堆叠
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff) for _ in range(num_encoder_layers)])
        
        # 通过 Decoder 层堆叠
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff) for _ in range(num_decoder_layers)])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        前向传播函数。

        Args:
            src (Tensor): 输入序列，形状为 (batch_size, src_len)。
            tgt (Tensor): 目标序列，形状为 (batch_size, tgt_len)。

        Returns:
            Tensor: 模型输出，形状为 (batch_size, tgt_len, vocab_size)。
        """
        
        # 1. 嵌入输入
        src = self.input_embedding(src) * self.scale
        tgt = self.output_embedding(tgt) * self.scale
        
        # 2. 添加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # 3. 编码器
        for layer in self.encoder:
            src = layer(src)
        
        # 4. 解码器
        for layer in self.decoder:
          # 这里decoder需要接受encoder的输出，以及自己的输入，两部分。图上可以看到，encoder有两根线连到decoder，同时decoder还有自己的输入。
            tgt = layer(tgt, src)
        
        # 5. 输出层
        output = self.fc_out(tgt)
        # 眼尖的读者会发现这里相比原论文，少了一层softmax。这是因为不需要显式添加了，我们后面会解释。
        return output
```


## 边缘模块
下面我们从好实现的边缘模块开始。
### Embedding
在我们的任务中，需要把人类的自然语言变成机器可以理解的东西。在这里我们使用的是一个一维向量。这个已经有库为我们实现好了，我们只要一行：
```python
    self.embedding = nn.Embedding(vocab_size, d_model)
```
就可以了。
### Positional Encoding
Embedding建立了单词到向量的映射。但是还有一个重要的信息就是顺序，我们说话的时候语序包含了很重要的信息，比如“我爱你”和“你爱我”是完全不同的意思。为了让模型知道单词的顺序，我们需要给每个单词添加一个位置编码。
在文章的第3.5字中提到位置编码的计算方法。我们可以使用正弦和余弦函数来计算位置编码。具体来说，对于每个位置pos和每个维度i，位置编码的计算公式如下：
$$
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
$$
$$
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
$$
其中，$d_{model}$是嵌入向量的维度，$pos$是单词在句子中的位置，$i$是嵌入向量的维度索引。这个公式的意思是，对于每个位置，我们使用正弦和余弦函数来计算位置编码。这样做的好处是，正弦和余弦函数是周期性的，可以捕捉到单词之间的相对位置关系。

然后我们直接把位置编码加到Embedding上面就可以了.
```Python
# positional_encoding.py

# 论文3.5节中提到了位置编码
import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # 生成一个形状为(max_len, 1)的张量，其中包含从0到max_len-1的值
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term就是分母10000^(2i/d_model)倒数，这里i是维度索引。因为embedding的维度是d_model，每个维度上都要有位置编码，不同维度不同，i就是维度。例如第0维的位置编码是10000^(2*0/d_model)。
        # 这个地方是Postional Encoding中最复杂的地方了，其实也没多复杂，就是想办法把让底数等于10000。因为exp(log(10000))=10000，那么exp(log(10000) * x) = 10000^x，就很容易了。这个式子的结果是10000^(-2i/d_model)。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 这是一个切片操作，第一个:表示选择所有行，第二个0::2表示从第0列开始，每隔2列选择一次。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 把pe的大小变成(1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 使pe成为模型的一部分，可以随设备转换。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假如x的大小是(batch_size, seq_len, d_model)，由于pe的大小是(1, max_len, d_model)，只要让pe的第一个维度和x一样，就可以直接加了。
        return x + self.pe[:, :x.size(1)]
```



## Attention
既然文章叫Attention Is All You Need，我们自然要看看Attention是什么。
## Encoder
我们人类说话时，有一种先理解，再生成的过程。比如当你听到别人问“吃了没”，我们会先接受到这些语音信号，然后在大脑中理解，明白对方是在问我们有没有吃饭。接着我们再生成对应的答案，吃了或者没吃。很自然的，在自然语言处理中，人们也设计了这两个过程，这就是Encoder-Decoder架构。

## Decoder


## 训练
这里我们假设读者和我们一样都还不知道模型是怎么训练的。我们这里再介绍一下。

### 准备数据