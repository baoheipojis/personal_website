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
既然文章叫Attention Is All You Need，我们自然要看看Attention是什么。在论文的3.2节中，作者详细介绍了Attention。这里我们帮助读者学习一下。    
在开始之前，先需要了解的一点是，Attention机制并不是这个文章提出的，它的意思正如标题，意思是“你只需要注意力”，它的创新点不是在于提出了注意力，而是发现了可以**只用**注意力，不用RNN。并且还提出了两个重要的概念：**缩放点积注意力**和**多头注意力**。下面我们介绍它们：
### 缩放点积注意力
论文的3.2.1提出了一种叫“缩放点积注意力”的东西。我们一个个看：
#### 点积
点积就是数量积啦。以防有人不知道什么是数量积。我们这里再简单介绍一下。点积就是两个向量的对应元素相乘，然后求和，得到一个标量。比如：
$$
a = [1, 2, 3]     \quad   
b = [4, 5, 6]
$$

则有
$$
a \cdot b = 1*4 + 2*5 + 3*6 = 32
$$

在文章的3.2.1公式下面那段就解释了，目前的注意力包括加性注意力和点积注意力这两种，以及为什么要用点积注意力。
#### 缩放
说到缩放前，我们需要提一个函数叫softmax，它的输入是一个n维向量，输出也是一个n维向量，用于把一个普通的向量转化成概率分布（所有数都在01之间且和为1）其中每个位置的值是：
$$
softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

通过缩放，我们可以使softmax的值不再接近0或1，从而缓解梯度消失的问题。这里采用的缩放方式是除以$\sqrt{d_k}$，其中$d_k$是key的维度。这是这个论文的贡献，之前有了点积注意力，但是没有缩放。
#### 公式
那么现在，我们就很容易理解这个公式（和3.2.1节给出的完全一样）了：
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
Q和K的维度都是(batch_size, seq_len, $d_k$)，$K^T$的维度是(batch_size,$d_k$, seq_len)，只转置最后两个维度。因此$QK^T$的维度是(batch_size, seq_len, seq_len)，缩放再softmax不影响维度。
V的维度是(batch_size, seq_len, $d_v$)，因此最后得到的结果是(batch_size, seq_len, $d_v$)，
### 多头注意力
接下来我们看看多头注意力，这是论文的3.2.2节提到的，也是首次提出的。
先看看论文说了一件什么事吧。初始状态下，Q,K,V矩阵的维度是一样的，都是(batch_size, seq_len, $d_model=512$)。$h=8,d_k=d_v=d_model/h=64$。我们使用8个头，每个头的处理是一样的：使用三个权重矩阵$W_i^Q,W_i^K,W_i^V$（显然，它们的维度应该是(512,64)，把Q,K,V从512维变成64维。然后根据上一节提到的缩放点积注意力，得到一个(batch_size, seq_len, $d_v=64$)的结果矩阵。8个头都得到一个64维的矩阵，最后拼起来就还是512维了。最后再乘一个权重矩阵$W^O$，得到最终的输出(batch_size, seq_len, $d_model=512$)。
### 代码实现
在代码实现中我们有一个变化就是，所有的头的权重矩阵合并起来，它们的大小都是(512,512)，是8个(512,64)权重矩阵拼起来的。思考一下是不是。Q是(b,s, 512)，我们把它和8个(512,64)的矩阵相乘得到的结果（每一个是(b,s,64)）再在最后一个维度上拼起来，效果和一个(512,512)的矩阵相乘是一样的。这我就不再多说明了，只要读者知道矩阵乘法是怎么做的就可以自行验证。
那么，最后得到的一个矩阵大小是(b,s,512)，其中最后一维是分成了8个头的，原本0号头就是在矩阵里的(b,s,0-63)，1号头在(b,s,64-127)以此类推。
接下来我们开始做缩放点积注意力，
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # 初始化参数，默认为512维，8个头
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # 这里权重矩阵就是(512,512)了。
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 分头处理
        Q = self.wq(q).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(k).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(v).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 最后变成了一个(b,nh,s,hd)的矩阵，把nh放到第二维，方便模拟处理多头。
        # 计算注意力分数。Q是(b,nh,s,hd)，K^T是(b,nh,hd,s)，乘起来是(b,nh,s,s)，和我们之前的分析一致。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # 把mask为0的位置的分数设为一个很小的值，-1e9。这样在softmax的时候就会被忽略掉。
            scores = scores.masked_fill(mask == 0, -1e9)
        # 在最后一个维度上做softmax
        attn = torch.softmax(scores, dim=-1)
        # V是(b,nh,s,hd)，attn是(b,nh,s,s)，乘起来是(b,nh,s,hd)
        output = torch.matmul(attn, V)
        
        # 现在我们要把output的维度从(b,8,s,64)变成(b,s,512)。
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.wo(output)
```
好啦，到这里，我们已经实现了这个论文里最核心的部分了，接下来我们看看其它部分吧。
## 位置前馈网络
下面我们来到3.3节，在这一节中提到了一个叫**位置前馈网络**(Position-wise Feed-Forward Networks)的东西，回到我们的Tranformer结构图，它就是那个蓝色的Feed Forward的部分。
它的输入x是Add&Norm的输出，大小是(b,s,d_model)。FFN的公式是：
$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$
很简单，就是线性层+ReLU+线性层。不过维度有点不同。我们这里的W1和W2的维度分别是(d_model, d_ff)和(d_ff, d_model)，d_ff=2048。
本质上是对每个位置进行的操作，只是我们合成矩阵方便并行化。我们来验算一下：
先让x乘上W1，得到的结果是(b,s,d_ff)，其中前两个位置固定下来时（例如0,0），最后的这个d_ff维的向量，就是x在(0,0)位置的向量乘上W1的结果。ReLU不影响维度。再乘上W2，同理可知，最后输出的相同位置的向量，还是只和x在(0,0)位置的向量有关。也就是对每个位置都做了同样的操作。
至于代码实现，那也很简单了
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

## Encoder
我们人类说话时，有一种先理解，再生成的过程。比如当你听到别人问“吃了没”，我们会先接受到这些语音信号，然后在大脑中理解，明白对方是在问我们有没有吃饭。接着我们再生成对应的答案，吃了或者没吃。很自然的，在自然语言处理中，人们也设计了这两个过程，这就是Encoder-Decoder架构。

再回到我们的Transformer结构图，Encoder包含Multi-Head Attention、Add&Norm、Feed Forward三个部分，其中Add&Norm并不复杂，我们马上就会讲到，其它两个的实现有点复杂，所以刚才单独说过了。


下面我们直接看代码实现，先看看Encoder层。
```python
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # 为什么2个norm，一个dropout呢。这是因为norm是有可学习参数的，两层norm需要区分开，但是dropout就无所谓了，即使是同一个，每次调用的结果也是不一样的。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 这里的attn怎么qkv输入都是x呢？这是因为，可学参数在权重W^Q,W^K,W^V上面，看结构图你也确实可以发现，multi-head attention这个层是由一个输入分成三份输入的。那为什么要用3个x当参数呢？既然都一样就直接用一个x，在attn里用3次不就行了？这是因为一会在decoder里会不一样的，别急。
        attn_output = self.self_attn(x, x, x, mask)
        # 这里是5.4提到的，应用dropout和残差连接
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```
Encoder就是把Encoder层给堆起来：


## Decoder
```Python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        # decoder有两层attention，一个是self attention，一个是cross attention。cross attention的接受encoder的输出。
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x
```

## 训练
这里我们假设读者和我们一样都还不知道模型是怎么训练的。我们这里再介绍一下。

### 准备数据