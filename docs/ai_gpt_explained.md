# z1_dart_gpt 详细解释：从 Dart 代码理解 AI 开发流程

本文解释 `bin/z1_dart_gpt.dart` 里的核心代码分别对应 AI 人工智能开发中的哪些步骤，以及每个概念是什么意思。

这份代码实现的是一个字符级 GPT。它不是调用现成深度学习框架，而是手写：

- 数据读取
- tokenizer
- 自动求导
- GPT 前向传播
- loss 计算
- 反向传播
- Adam 优化器
- 文本生成

## 总览：代码步骤等于 AI 开发步骤

| 代码阶段 | AI 开发阶段 | 这个阶段在做什么 |
| --- | --- | --- |
| 读取本地 `input.txt` | 数据准备 | 加载模型要学习的样本 |
| 构造 `uChars`、`charToId` | tokenizer / 特征编码 | 把字符映射成整数 token |
| 创建 `Value` | 自动求导系统 | 记录计算过程，让模型可以学习 |
| 创建 `stateDict` | 模型参数初始化 | 准备可训练参数 |
| `gpt()` | 模型架构 / 前向传播 | 根据上下文预测下一个 token |
| `softmax()` | 输出概率化 | 把模型分数变成概率 |
| `loss = -log(prob)` | 损失函数 | 衡量预测离正确答案有多远 |
| `loss.backward()` | 反向传播 | 计算每个参数对错误的责任 |
| Adam 更新 | 优化器 | 按梯度调整参数 |
| `runInference()` | 推理 / 生成 | 用训练后的模型生成新名字 |

## 1. 数据集：`input.txt`

代码：

```dart
final docs = File('input.txt')
    .readAsLinesSync()
    .map((line) => line.trim())
    .where((line) => line.isNotEmpty)
    .toList();
```

这一步等于 AI 开发中的“数据准备”。

`input.txt` 里每一行是一个名字，例如：

```text
王伟
李若然
张明宇
```

模型看到的不是“规则”，而是大量样本。它通过样本统计出：哪些字符经常出现，哪些字符组合更像名字，什么位置更可能出现什么字符。

在真实 AI 项目中，这一步可能对应：

- 收集文本、图片、音频或日志
- 清洗脏数据
- 去重
- 过滤异常样本
- 划分训练集和验证集

在这个项目里，数据非常简单：项目根目录的本地 `input.txt` 里一行一个名字。程序不会联网下载数据，缺少文件时会直接报错提醒你创建。

## 2. Tokenizer：把文字变成数字

代码：

```dart
uChars = docs.expand(charsOf).toSet().toList()..sort();
bos = uChars.length;
vocabSize = uChars.length + 1;
final charToId = {for (var i = 0; i < uChars.length; i++) uChars[i]: i};
```

这一步等于 AI 开发中的“tokenization”。

神经网络不能直接处理字符串。它只能处理数字。所以必须先把字符变成整数。

例如中文姓名数据里的字符集合可能包含：

```text
王 李 张 刘 伟 若 然 明 宇 ...
```

那么可能得到：

```text
王 -> 0
李 -> 1
张 -> 2
...
BOS -> 最后一个 token id
```

这里的 `BOS` 是 Beginning Of Sequence，意思是“序列开始”。代码也把它用作结束符。训练时一个名字会变成：

```text
王伟
```

对应 token 序列：

```text
[BOS, 王, 伟, BOS]
```

为什么前后都加 BOS？

- 第一个 BOS 表示“现在要开始生成一个名字”。
- 最后一个 BOS 表示“这个名字结束了”。

真实大模型也有 tokenizer，只是通常不是字符级，而是子词级。中文词语、英文单词、标点、空格片段都可能被编码成 token。

## 3. `Value`：最小自动求导系统

代码里的 `Value` 是整个项目最核心的底层结构之一。

```dart
class Value {
  double data;
  double grad = 0.0;
  final List<Value> _children;
  final List<double> _localGrads;
}
```

它等于 AI 开发中的“自动求导引擎”。

每个 `Value` 表示一个标量数字，并且记录：

- `data`：这个节点的数值
- `grad`：最终 loss 对这个节点的导数
- `_children`：这个值由哪些旧值计算而来
- `_localGrads`：当前运算对每个输入的局部导数

例如：

```text
z = x * y
```

那么：

```text
dz/dx = y
dz/dy = x
```

代码里乘法就是这样写的：

```dart
return Value(data * o.data, <Value>[this, o], <double>[o.data, data]);
```

这句话的意思是：

- 新值等于 `this.data * o.data`
- 新值依赖两个输入：`this` 和 `o`
- 对第一个输入的局部导数是 `o.data`
- 对第二个输入的局部导数是 `this.data`

## 4. `backward()`：反向传播是什么意思

代码：

```dart
loss.backward();
```

这一步等于 AI 开发中的“backpropagation”，也就是反向传播。

训练模型时，我们先得到一个 loss。loss 越大，说明模型预测越差。问题是：模型里有很多参数，应该调哪个？调多少？

反向传播做的事情就是：

```text
计算 loss 对每个参数的导数
```

导数可以理解为“这个参数往哪个方向变，会让 loss 下降”。

`backward()` 内部先做拓扑排序：

```dart
void buildTopo(Value v) {
  if (visited.add(v)) {
    for (final child in v._children) {
      buildTopo(child);
    }
    topo.add(v);
  }
}
```

因为一个复杂计算图里，后面的值依赖前面的值。反向传播必须按正确顺序，从 loss 一路往回传。

然后：

```dart
child.grad += localGrad * v.grad;
```

这就是链式法则：

```text
loss 对 child 的影响
= loss 对当前节点的影响 * 当前节点对 child 的影响
```

真实框架里的 PyTorch `loss.backward()` 做的也是这个思想，只是它处理的是大张量和 GPU 计算。

## 5. 模型参数：`stateDict`

代码：

```dart
final stateDict = <String, List<List<Value>>>{};
```

以及：

```dart
stateDict['wte'] = matrix(vocabSize, nEmbd);
stateDict['wpe'] = matrix(blockSize, nEmbd);
stateDict['lm_head'] = matrix(vocabSize, nEmbd);
```

这一步等于 AI 开发中的“模型参数初始化”。

参数就是模型会学习的数字。训练开始前，它们是随机的；训练过程中，优化器不断修改它们；训练结束后，它们保存了模型从数据中学到的统计规律。

这个项目里的参数包括：

| 参数名 | 含义 |
| --- | --- |
| `wte` | token embedding，把 token id 映射成向量 |
| `wpe` | position embedding，表示当前位置 |
| `lm_head` | 输出层，把隐藏向量映射回词表 logits |
| `attn_wq` | attention query 权重 |
| `attn_wk` | attention key 权重 |
| `attn_wv` | attention value 权重 |
| `attn_wo` | attention 输出投影 |
| `mlp_fc1` | MLP 第一层 |
| `mlp_fc2` | MLP 第二层 |

`params` 是把所有矩阵拍平成一个列表：

```dart
params = [
  for (final mat in stateDict.values)
    for (final row in mat)
      for (final p in row) p,
];
```

这样 Adam 优化器就可以统一遍历所有参数。

## 6. Embedding：token 变向量

代码：

```dart
final tokEmb = stateDict['wte']![tokenId];
final posEmb = stateDict['wpe']![posId];
var x = [for (var i = 0; i < nEmbd; i++) tokEmb[i] + posEmb[i]];
```

这一步等于 AI 开发中的“embedding lookup”。

token id 本身只是一个整数，比如 `a = 0`。整数之间的大小没有自然语义。模型需要把 token 转成向量。

例如：

```text
token id: 0
embedding: [0.03, -0.12, 0.07, ...]
```

`tokEmb` 表示字符本身的含义。`posEmb` 表示它在序列中的位置。

为什么需要位置？

因为 attention 本身并不知道顺序。如果没有位置编码，模型很难区分：

```text
abc
cba
```

## 7. `gpt()`：模型前向传播

代码：

```dart
final logits = gpt(tokenId, posId, keys, values);
```

这一步等于 AI 开发中的“forward pass”，也就是前向传播。

前向传播的意思是：

```text
输入当前 token 和历史上下文，计算下一个 token 的预测分数
```

`gpt()` 返回的是 `logits`。

logits 是还没有归一化的分数。例如：

```text
a: 1.2
b: -0.3
c: 0.8
```

分数越高，模型越认为下一个字符可能是它。

## 8. Attention：让模型看历史上下文

代码：

```dart
final q = linear(x, stateDict['layer$li.attn_wq']!);
final k = linear(x, stateDict['layer$li.attn_wk']!);
final v = linear(x, stateDict['layer$li.attn_wv']!);
```

attention 是 GPT 的核心。

可以把它理解成：

```text
当前字符在预测下一个字符时，应该重点参考前面哪些字符？
```

例如生成名字：

```text
王明
```

模型要预测下一个字符，它可能需要参考：

- `王`：姓氏信息
- `明`：当前最靠近的位置，也暗示后面可能接 `宇`、`轩`、`泽` 等名字用字

Query、Key、Value 的意思：

| 名称 | 简写 | 含义 |
| --- | --- | --- |
| Query | Q | 当前 token 发出的“我要找什么信息” |
| Key | K | 历史 token 提供的“我有什么特征” |
| Value | V | 历史 token 真正贡献给输出的信息 |

代码里通过：

```dart
dot(qH, kH[t]) / math.sqrt(headDim)
```

计算当前 query 和历史 key 的相似度。相似度越高，说明当前 token 越应该关注那个历史 token。

然后：

```dart
final attnWeights = softmax(attnLogits);
```

把相似度变成权重。最后用权重加权 value：

```dart
attnWeights[t] * vH[t][j]
```

这就是 attention 的核心计算。

## 9. Multi-head Attention：多个角度看上下文

代码：

```dart
const nHead = 4;
const headDim = nEmbd ~/ nHead;
```

multi-head attention 的意思是：

```text
不是只用一个注意力视角，而是分成多个头，从多个角度观察上下文
```

在这个代码里：

```text
nEmbd = 16
nHead = 4
headDim = 4
```

也就是 16 维向量被分成 4 个头，每个头处理 4 维。

一个头可能更关注姓氏，一个头可能更关注最近字符，一个头可能更关注名字用字组合。这个解释是直观理解，不是代码里手动指定的规则；具体关注什么，是训练学出来的。

## 10. RMSNorm：稳定数值

代码：

```dart
List<Value> rmsnorm(List<Value> x) {
  final ms = sumValues([for (final xi in x) xi * xi]) / x.length;
  final scale = (ms + 1e-5).pow(-0.5);
  return [for (final xi in x) xi * scale];
}
```

这一步等于 AI 开发中的“normalization”。

神经网络每一层的数值如果变得太大或太小，训练会不稳定。RMSNorm 会把向量缩放到更稳定的范围。

这里的逻辑是：

1. 计算平方均值 `ms`
2. 计算缩放系数 `scale`
3. 每个元素乘以 `scale`

`1e-5` 是防止除零或数值过小的稳定项。

## 11. MLP：每个位置上的非线性加工

代码：

```dart
x = linear(x, stateDict['layer$li.mlp_fc1']!);
x = [for (final xi in x) xi.relu()];
x = linear(x, stateDict['layer$li.mlp_fc2']!);
```

MLP 是 feed-forward network，也叫前馈网络。

attention 负责“从上下文取信息”，MLP 负责“对当前位置的信息做加工”。

这里的 MLP 做了：

```text
16 维 -> 64 维 -> ReLU -> 16 维
```

`ReLU` 是激活函数：

```text
relu(x) = max(0, x)
```

没有激活函数，很多层线性变换叠在一起仍然只是线性变换，表达能力会弱很多。

## 12. Residual：残差连接

代码：

```dart
x = [for (var i = 0; i < x.length; i++) x[i] + xResidual[i]];
```

这一步等于 AI 开发中的“residual connection”。

残差连接的意思是：

```text
当前层不要完全覆盖输入，而是在输入基础上加上新学到的变化
```

好处是：

- 梯度更容易往前传
- 深层模型更稳定
- 模型可以选择“少改一点”或“多改一点”

GPT、ResNet 等很多现代神经网络都大量使用残差连接。

## 13. Softmax：分数变概率

代码：

```dart
final probs = softmax(logits);
```

`logits` 是原始分数，`softmax` 把它变成概率。

例如：

```text
logits: [2.0, 1.0, 0.1]
probs:  [0.66, 0.24, 0.10]
```

概率的特点是：

- 每个值大于等于 0
- 所有值加起来等于 1

这样模型就能表达：

```text
下一个字符是 a 的概率是多少
下一个字符是 b 的概率是多少
...
```

## 14. Loss：模型错得有多严重

代码：

```dart
losses.add(-probs[targetId].log());
```

这一步等于 AI 开发中的“损失函数”。

`targetId` 是正确答案。如果模型给正确答案的概率很高，loss 就低；如果模型给正确答案的概率很低，loss 就高。

例如：

```text
正确答案概率 = 0.9
-log(0.9) 很小

正确答案概率 = 0.01
-log(0.01) 很大
```

这就是交叉熵损失的核心形式。

训练目标不是直接“生成好名字”，而是让 loss 变小。loss 变小之后，模型自然更擅长预测下一个字符。

## 15. Adam：参数怎么更新

代码：

```dart
m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad;
v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad * p.grad;
p.data -= lrT * mHat / (math.sqrt(vHat) + epsAdam);
```

这一步等于 AI 开发中的“optimizer”。

优化器负责根据梯度更新参数。

最简单的梯度下降是：

```text
参数 = 参数 - 学习率 * 梯度
```

Adam 更聪明一点，它会维护两个统计量：

| 变量 | 意思 |
| --- | --- |
| `m` | 梯度的一阶动量，可以理解为梯度的滑动平均 |
| `v` | 梯度平方的二阶动量，可以理解为梯度大小的滑动平均 |

Adam 的效果通常比普通 SGD 更稳定，尤其适合很多深度学习任务。

## 16. Training loop：完整训练循环

代码：

```dart
for (var step = 0; step < numSteps; step++) {
  ...
}
```

每一步训练做的事情是：

1. 取一个名字。
2. 加上开头和结尾 BOS。
3. 从左到右预测每个下一个字符。
4. 累加每个位置的 loss。
5. 对平均 loss 做反向传播。
6. 用 Adam 更新所有参数。
7. 清空梯度。

这对应真实 AI 训练里的标准流程：

```text
batch -> forward -> loss -> backward -> optimizer step -> zero grad
```

区别是这里一次只训练一个名字，不做 batch。这样代码更短，但速度更慢、训练也更不稳定。

## 17. Inference：训练后怎么生成

代码：

```dart
void runInference() {
  const temperature = 0.5;
  ...
}
```

这一步等于 AI 开发中的“推理”或“生成”。

训练时，模型知道正确答案，所以可以计算 loss。推理时，没有正确答案，模型只能自己一步一步生成。

流程是：

1. 从 `BOS` 开始。
2. 调用 `gpt()` 得到下一个字符概率。
3. 按概率采样一个字符。
4. 把这个字符加入结果。
5. 继续预测下一个字符。
6. 如果采样到 `BOS`，说明名字结束。

这和聊天模型生成回答的方式非常像。聊天模型也是先有一段上下文，然后不断预测下一个 token。

## 18. Temperature：生成时的随机程度

代码：

```dart
const temperature = 0.5;
final probs = softmax([for (final l in logits) l / temperature]);
```

temperature 控制生成的随机性。

当 temperature 较低时：

- 高概率 token 更容易被选中
- 输出更稳定
- 但可能更重复、更保守

当 temperature 较高时：

- 低概率 token 也更有机会被选中
- 输出更多样
- 但可能更奇怪

这里用 `0.5`，偏保守，所以生成的名字更像训练数据。

## 19. 这些变量名分别是什么意思

| 名称 | 全称或含义 | 解释 |
| --- | --- | --- |
| `docs` | documents | 训练样本列表 |
| `uChars` | unique characters | 数据集中出现过的所有字符 |
| `BOS` / `bos` | Beginning Of Sequence | 序列开始标记，也用作结束标记 |
| `vocabSize` | vocabulary size | token 总数 |
| `nLayer` | number of layers | Transformer 层数 |
| `nEmbd` | embedding dimension | 向量宽度 |
| `blockSize` | context length | 最长上下文长度 |
| `nHead` | number of attention heads | 注意力头数量 |
| `headDim` | head dimension | 每个注意力头的维度 |
| `stateDict` | state dictionary | 保存所有模型参数的表 |
| `wte` | word/token embedding | token embedding 矩阵 |
| `wpe` | word/position embedding | position embedding 矩阵 |
| `lm_head` | language model head | 输出层 |
| `q` | query | 当前 token 要查询什么 |
| `k` | key | 历史 token 有什么特征 |
| `v` | value | 历史 token 提供什么内容 |
| `logits` | raw scores | softmax 前的原始分数 |
| `probs` | probabilities | softmax 后的概率 |
| `loss` | loss value | 模型错误程度 |
| `grad` | gradient | loss 对某个值的导数 |
| `lrT` | learning rate at step t | 当前步学习率 |

## 20. 为什么这是 GPT

GPT 的核心特征是：

- 只看过去，不看未来
- 根据上下文预测下一个 token
- 使用 Transformer decoder 风格结构
- 训练目标是 next-token prediction

这份代码满足这些特点：

- `keys` 和 `values` 只保存已经出现过的位置。
- 每一步只预测 `tokens[posId + 1]`。
- `gpt()` 里有 attention、MLP、norm、residual。
- 训练目标是下一个字符的负对数似然。

所以它是一个极小、字符级、教学版 GPT。

## 21. 和真实大模型有什么不同

真实 GPT 模型会大很多，也复杂很多。

| 本项目 | 真实大模型 |
| --- | --- |
| 字符级 tokenizer | 子词级 tokenizer |
| 13824 个参数左右 | 数十亿到数万亿参数 |
| 标量自动求导 | 张量自动求导 |
| CPU 单线程 | GPU/TPU 大规模并行 |
| 一次训练一个名字 | 大 batch 训练 |
| 只有 1 层 | 几十到上百层 |
| ReLU | 常见为 GeLU / SwiGLU 等 |
| 教学用途 | 生产或研究用途 |

但是核心思想是一样的：

```text
数据 -> token -> 模型 -> loss -> 梯度 -> 参数更新 -> 推理生成
```

## 22. 如何阅读这份代码

推荐阅读顺序：

1. 先看 `main()`，理解整体流程。
2. 看 tokenizer 相关的 `uChars`、`bos`、`charToId`。
3. 看 `Value`，理解自动求导。
4. 看 `initStateDict()`，理解参数有哪些。
5. 看 `gpt()`，理解模型结构。
6. 看 `train()`，理解训练循环。
7. 看 `runInference()`，理解生成过程。

如果只想抓住一句话：

```text
这份代码训练了一个小模型，让它根据已经看到的字符预测下一个字符，然后用同样的预测能力生成新名字。
```
