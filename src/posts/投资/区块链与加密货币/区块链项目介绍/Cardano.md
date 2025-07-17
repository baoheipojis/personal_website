# Cardano

## 共识机制
下面，我们介绍Cardano的共识机制，这是第一个经过形式化验证的共识机制，Ouroboros。
先来回顾一下共识机制需要干嘛
### Ouroboros
Ouroboros是Cardano的共识机制。
### 长程攻击
我们知道PoW有51%攻击，当然，PoS也有51%攻击，原理完全相同，这里不再多说。下面主要介绍一种PoS独有的攻击方式——长程攻击（long-range attack）。
一般来说，PoS通过“检查点”（checkpoint）来防止长程攻击。检查点机制的问题在于新加入的节点必须信任某个中心化的检查点，但是Cardano不需要！
当然，Cardano的去中心化也饱受批评，最重要的一点是Cardano的开发团队IOHK是一个中心化的公司。

