# TS2Vec

## Abstract

TS2Vec 在增强的上下文视图上以分层方式进行对比学习，这为每个时间戳提供了强大的上下文表示。

> 在 TimeMAE 中揭示了这种方法的问题：
>
> > 当前工作中，对比学习范式最为流行。对比学习的共同范式是学习 embedding，并且假设这些 embedding 对各种规模输入的扭曲具有不变性。
> >
> > 这种方法有很多缺点：
> >
> > - 尽管这些方法很流行，但是**不变性假设在现实世界中可能不总是成立**。并且它在开发数据增强策略中带来太多归纳偏差，并且在负采样中引入额外的偏差。
> >
> > - 更大的缺点是：它们本质上使用**单向编码器的（unidirectional encoder）**方案来学习时间序列的表示，限制了**上下文表示（contextual representations）**的提取。
>
> 这种问题，目前有很好的方式来解决，也就是掩码自编码器（Masked AutoEncoder）



## Code

1. `a[:, train_slice]`等同于在所有维度上应用切片，但只为前两维提供了具体的切片指令。第三维（如果存在）将被隐式地完整包含，就像你写了`a[:, train_slice, :]`一样

2. ```python
   labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
   ```

   - data的原始大小为：`[1, n, 7]`

   - 每个切片的大小是：`[1, n - pred_len + 1, 7]`，切片的范围从`[0: n - pred_len]`（含stop） 到 `[pred_len - 1, n - 1]`（含 stop）

     其中 `[0: n - pred_len]` 的数据都要预测，预测其后 pred_len 个数据点。但是 0 数据点对应的 `[0: pred_len - 1]` 最后是被删去的，因为其前面没有数据，不可以预测。

   - 堆叠 `pred_len` 个这样的数组，会在新的第三维上增加一个维度，所以堆叠后的形状变为`[1, n - pred_len + 1, pred_len, 7]`。

   - 移除每个切片中的第一个时间步：第二维（时间维）上的长度从 `n+1-pred_len` 变为 `n-pred_len`（移除了第一个元素），因此最终的形状变为`[1, n-pred_len, pred_len, 7]`。

     因此 `[1: n - pred_len]` 中每个时间步都要预测之后 pred_len 长度的数据。

3. ```python
   features = features[:, :-pred_len]
   ```

   features 中 `[0: n - pred_len - 1, n_dims]` 被保留下来

4. ```python
   return features.reshape(-1, features.shape[-1]), \
          labels.reshape(-1, labels.shape[2]*labels.shape[3])
   ```

   features 之前的规模：`[bs, 0: n - pred_len - 1, n_dims]`，现在变成 `[bs * (0: n - pred_len - 1): n_dims]`。

   labels 之前的规模：`[bs, 1: n - pred_len, pred_len, n_features]`，现在变成 `[bs * (1: n - pred_len), pred_len * n_features]`。

   也就是由 0 预测 `[1: pred_len]`

   由 `0, 1` 预测 `[2: pred_len - 1]`

   由 `0, 1, …, n - pred_len - 1` 预测 `[n - pred_len: n - 1]`

   features 是  `0, 1, …, n - pred_len - 1` 这 `n - pred_len` 个点的表征

   labels 是对应预测结果的真实值。

## Word

hierarchical：分层的

augmented：增强的

context：上下文

anomaly：异常