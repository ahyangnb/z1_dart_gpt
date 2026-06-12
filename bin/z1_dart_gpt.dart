import 'dart:collection';
import 'dart:io';
import 'dart:math' as math;

// 固定种子，便于复现实验结果。
final rng = math.Random(42);

const nLayer = 1;
const nEmbd = 16;
const blockSize = 16;
const nHead = 4;
const headDim = nEmbd ~/ nHead;

late List<String> uChars;
late int bos;
late int vocabSize;
late List<Value> params;

final stateDict = <String, List<List<Value>>>{};

// 标量自动求导节点：保存数值、梯度和局部导数。
class Value {
  Value(num data, [List<Value>? children, List<double>? localGrads])
      : data = data.toDouble(),
        _children = children ?? const <Value>[],
        _localGrads = localGrads ?? const <double>[];

  double data;
  double grad = 0.0;
  final List<Value> _children;
  final List<double> _localGrads;

  Value operator +(Object other) {
    final o = _asValue(other);
    return Value(data + o.data, <Value>[this, o], <double>[1.0, 1.0]);
  }

  Value operator *(Object other) {
    final o = _asValue(other);
    return Value(data * o.data, <Value>[this, o], <double>[o.data, data]);
  }

  Value operator -() => this * -1.0;

  Value operator -(Object other) => this + (-_asValue(other));

  Value operator /(Object other) => this * _asValue(other).pow(-1.0);

  Value pow(double exponent) {
    final out = math.pow(data, exponent).toDouble();
    final localGrad = exponent * math.pow(data, exponent - 1).toDouble();
    return Value(out, <Value>[this], <double>[localGrad]);
  }

  Value log() => Value(math.log(data), <Value>[this], <double>[1.0 / data]);

  Value exp() {
    final out = math.exp(data);
    return Value(out, <Value>[this], <double>[out]);
  }

  Value relu() {
    return Value(
      math.max(0.0, data),
      <Value>[this],
      <double>[data > 0.0 ? 1.0 : 0.0],
    );
  }

  // 反向传播：拓扑排序后按链式法则累加梯度。
  void backward() {
    final topo = <Value>[];
    final visited = HashSet<Value>.identity();

    void buildTopo(Value v) {
      if (visited.add(v)) {
        for (final child in v._children) {
          buildTopo(child);
        }
        topo.add(v);
      }
    }

    buildTopo(this);
    grad = 1.0;

    for (final v in topo.reversed) {
      for (var i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * v.grad;
      }
    }
  }
}

Value _asValue(Object other) {
  if (other is Value) return other;
  if (other is num) return Value(other);
  throw ArgumentError.value(other, 'other', 'Expected a Value or num.');
}

// 主流程：准备数据、初始化模型、训练并采样。
void main() {
  ensureInputFile();

  final docs = File('input.txt')
      .readAsLinesSync()
      .map((line) => line.trim())
      .where((line) => line.isNotEmpty)
      .toList();
  if (docs.isEmpty) {
    throw StateError('input.txt is empty. Add one training sample per line.');
  }
  docs.shuffle(rng);
  print('num docs: ${docs.length}');

  uChars = docs.expand(charsOf).toSet().toList()..sort();
  bos = uChars.length;
  vocabSize = uChars.length + 1;
  final charToId = {for (var i = 0; i < uChars.length; i++) uChars[i]: i};
  print('vocab size: $vocabSize');

  initStateDict();
  params = [
    for (final mat in stateDict.values)
      for (final row in mat)
        for (final p in row) p,
  ];
  print('num params: ${params.length}');

  final numSteps = intFromEnvironment('Z1_DART_GPT_STEPS', 1000);
  train(docs, charToId, numSteps);
  runInference();
}

void ensureInputFile() {
  final file = File('input.txt');
  if (!file.existsSync()) {
    throw StateError(
      'Missing input.txt. Create input.txt in the project root, one sample per line.',
    );
  }
}

int intFromEnvironment(String name, int defaultValue) {
  final raw = Platform.environment[name];
  if (raw == null) return defaultValue;
  final parsed = int.tryParse(raw);
  return parsed == null || parsed < 1 ? defaultValue : parsed;
}

List<String> charsOf(String s) {
  return s.runes.map(String.fromCharCode).toList();
}

double randomGaussian(double mean, double std) {
  var u1 = 0.0;
  while (u1 == 0.0) {
    u1 = rng.nextDouble();
  }
  final u2 = rng.nextDouble();
  final z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  return mean + std * z0;
}

// 参数矩阵用小的高斯噪声初始化。
List<List<Value>> matrix(int nout, int nin, [double std = 0.08]) {
  return List.generate(
    nout,
    (_) => List.generate(nin, (_) => Value(randomGaussian(0.0, std))),
  );
}

void initStateDict() {
  stateDict['wte'] = matrix(vocabSize, nEmbd);
  stateDict['wpe'] = matrix(blockSize, nEmbd);
  stateDict['lm_head'] = matrix(vocabSize, nEmbd);

  for (var i = 0; i < nLayer; i++) {
    stateDict['layer$i.attn_wq'] = matrix(nEmbd, nEmbd);
    stateDict['layer$i.attn_wk'] = matrix(nEmbd, nEmbd);
    stateDict['layer$i.attn_wv'] = matrix(nEmbd, nEmbd);
    stateDict['layer$i.attn_wo'] = matrix(nEmbd, nEmbd);
    stateDict['layer$i.mlp_fc1'] = matrix(4 * nEmbd, nEmbd);
    stateDict['layer$i.mlp_fc2'] = matrix(nEmbd, 4 * nEmbd);
  }
}

List<Value> linear(List<Value> x, List<List<Value>> w) {
  return [for (final row in w) dot(row, x)];
}

Value dot(List<Value> a, List<Value> b) {
  if (a.isEmpty) return Value(0.0);
  var out = a[0] * b[0];
  for (var i = 1; i < a.length; i++) {
    out = out + (a[i] * b[i]);
  }
  return out;
}

Value sumValues(Iterable<Value> values) {
  final iterator = values.iterator;
  if (!iterator.moveNext()) return Value(0.0);

  var total = iterator.current;
  while (iterator.moveNext()) {
    total = total + iterator.current;
  }
  return total;
}

List<Value> softmax(List<Value> logits) {
  final maxVal = logits.map((val) => val.data).reduce(math.max);
  final exps = [for (final val in logits) (val - maxVal).exp()];
  final total = sumValues(exps);
  return [for (final e in exps) e / total];
}

List<Value> rmsnorm(List<Value> x) {
  final ms = sumValues([for (final xi in x) xi * xi]) / x.length;
  final scale = (ms + 1e-5).pow(-0.5);
  return [for (final xi in x) xi * scale];
}

// 单步 GPT 前向：输入当前 token，输出下一个 token 的 logits。
List<Value> gpt(
  int tokenId,
  int posId,
  List<List<List<Value>>> keys,
  List<List<List<Value>>> values,
) {
  final tokEmb = stateDict['wte']![tokenId];
  final posEmb = stateDict['wpe']![posId];
  var x = [for (var i = 0; i < nEmbd; i++) tokEmb[i] + posEmb[i]];
  x = rmsnorm(x);
  // 每层包含注意力块和 MLP 块，各自带残差连接。
  for (var li = 0; li < nLayer; li++) {
    final xResidual = x;
    x = rmsnorm(x);

    final q = linear(x, stateDict['layer$li.attn_wq']!);
    final k = linear(x, stateDict['layer$li.attn_wk']!);
    final v = linear(x, stateDict['layer$li.attn_wv']!);
    keys[li].add(k);
    values[li].add(v);
    // 用历史 key/value 缓存实现因果注意力。
    final xAttn = <Value>[];
    for (var h = 0; h < nHead; h++) {
      final hs = h * headDim;
      final qH = q.sublist(hs, hs + headDim);
      final kH = [for (final ki in keys[li]) ki.sublist(hs, hs + headDim)];
      final vH = [for (final vi in values[li]) vi.sublist(hs, hs + headDim)];

      final attnLogits = [
        for (var t = 0; t < kH.length; t++) dot(qH, kH[t]) / math.sqrt(headDim),
      ];
      final attnWeights = softmax(attnLogits);

      final headOut = [
        for (var j = 0; j < headDim; j++)
          sumValues([
            for (var t = 0; t < vH.length; t++) attnWeights[t] * vH[t][j],
          ]),
      ];
      xAttn.addAll(headOut);
    }

    x = linear(xAttn, stateDict['layer$li.attn_wo']!);
    x = [for (var i = 0; i < x.length; i++) x[i] + xResidual[i]];
    // 前馈网络：RMSNorm -> Linear -> ReLU -> Linear -> Residual。
    final mlpResidual = x;
    x = rmsnorm(x);
    x = linear(x, stateDict['layer$li.mlp_fc1']!);
    x = [for (final xi in x) xi.relu()];
    x = linear(x, stateDict['layer$li.mlp_fc2']!);
    x = [for (var i = 0; i < x.length; i++) x[i] + mlpResidual[i]];
  }

  return linear(x, stateDict['lm_head']!);
}

// 训练循环：逐个名字构造序列，最小化下一个字符损失。
void train(List<String> docs, Map<String, int> charToId, int numSteps) {
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;

  final m = List.filled(params.length, 0.0);
  final v = List.filled(params.length, 0.0);

  for (var step = 0; step < numSteps; step++) {
    final doc = docs[step % docs.length];
    final tokens = <int>[
      bos,
      for (final ch in charsOf(doc)) charToId[ch]!,
      bos,
    ];
    final n = math.min(blockSize, tokens.length - 1);

    final keys = List.generate(nLayer, (_) => <List<Value>>[]);
    final values = List.generate(nLayer, (_) => <List<Value>>[]);
    final losses = <Value>[];

    for (var posId = 0; posId < n; posId++) {
      final tokenId = tokens[posId];
      final targetId = tokens[posId + 1];
      final logits = gpt(tokenId, posId, keys, values);
      final probs = softmax(logits);
      losses.add(-probs[targetId].log());
    }

    final loss = sumValues(losses) * (1.0 / n);
    loss.backward();
    // Adam 更新参数后清零梯度，进入下一步。
    final lrT = learningRate * (1.0 - step / numSteps);
    for (var i = 0; i < params.length; i++) {
      final p = params[i];
      m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad * p.grad;

      final mHat = m[i] / (1.0 - math.pow(beta1, step + 1));
      final vHat = v[i] / (1.0 - math.pow(beta2, step + 1));
      p.data -= lrT * mHat / (math.sqrt(vHat) + epsAdam);
      p.grad = 0.0;
    }

    stdout.write(
      'step ${(step + 1).toString().padLeft(4)} / '
      '${numSteps.toString().padLeft(4)} | '
      'loss ${loss.data.toStringAsFixed(4)}\r',
    );
  }
}

// 推理阶段从 BOS 开始，按概率采样直到再次生成 BOS。
void runInference() {
  const temperature = 0.5;
  print('\n--- 推理结果（生成的中文名字）---');

  for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    final keys = List.generate(nLayer, (_) => <List<Value>>[]);
    final values = List.generate(nLayer, (_) => <List<Value>>[]);
    var tokenId = bos;
    final sample = <String>[];

    for (var posId = 0; posId < blockSize; posId++) {
      final logits = gpt(tokenId, posId, keys, values);
      final probs = softmax([for (final l in logits) l / temperature]);
      tokenId = sampleWeighted([for (final p in probs) p.data]);
      if (tokenId == bos) break;
      sample.add(uChars[tokenId]);
    }

    print('样本 ${(sampleIdx + 1).toString().padLeft(2)}: ${sample.join()}');
  }
}

// 简单的按权重采样，输入是 softmax 后的概率。
int sampleWeighted(List<double> weights) {
  final total = weights.fold(0.0, (sum, weight) => sum + weight);
  var draw = rng.nextDouble() * total;

  for (var i = 0; i < weights.length; i++) {
    draw -= weights[i];
    if (draw <= 0.0) return i;
  }
  return weights.length - 1;
}
