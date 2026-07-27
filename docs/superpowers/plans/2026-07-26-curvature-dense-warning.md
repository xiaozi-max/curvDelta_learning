# DenseModel(BaseTuner) 完整实现清单（peft 0.17.1）

## 核心原则

`BaseTuner`（`peft.tuners.tuners_utils.BaseTuner`）是一个**几乎不提供默认实现的抽象基类**。它只负责调度（`inject_adapter` 遍历模型、匹配 `target_modules`、调用你的钩子），具体怎么替换模块、怎么管理 adapter 状态，全部要你自己写。这一点从 0.13 到 main 分支一直没变，**不是版本差异**，是 `BaseTuner` 的设计方式——`LoraModel`/`IA3Model` 能用,是因为它们把下面这 9 个方法都补齐了。

分两类：

- **6 个 `@abstractmethod`**：漏了任何一个，`class DenseModel(BaseTuner): ...` 这行代码本身就会在类定义时报 `TypeError: Can't instantiate abstract class`，不会等到运行时才炸。
- **3 个非 abstract 但实践必需**：类定义能过，但一运行就在具体调用点报 `AttributeError`。这三个最容易被漏掉，因为 IDE/linter 不会提示。

---

## 第一类：6 个 abstractmethod（必须实现，否则类定义即失败）

### 1. `_prepare_adapter_config(self, peft_config, model_config) -> PeftConfig`

作用：`target_modules` 为空时，按模型类型自动填充。`LoraModel`（v0.17.0）里这个是 `@staticmethod`，你也应该照抄成 staticmethod（虽然基类签名是实例方法，但 Python 允许子类用 staticmethod 覆盖，`LoraModel` 就是这么做的，实测无问题）：

```python
@staticmethod
def _prepare_adapter_config(peft_config, model_config):
    if peft_config.target_modules is None:
        target_modules = DenseModel.target_module_mapping.get(
            model_config.get("model_type", ""), []
        )
        if target_modules:
            peft_config.target_modules = (
                set(target_modules) if not isinstance(target_modules, str) else target_modules
            )
        else:
            raise ValueError("Please specify `target_modules` in `peft_config`.")
    return peft_config
```

### 2. `_check_target_module_exists(peft_config, key) -> bool`

同样是 staticmethod（对齐 `LoraModel`），直接复用 peft 提供的通用正则匹配工具，不用自己写：

```python
from peft.tuners.tuners_utils import check_target_module_exists

@staticmethod
def _check_target_module_exists(dense_config, key):
    return check_target_module_exists(dense_config, key)
```

### 3. `_create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs)`

基类抽象签名（0.17.1 的 `icedit` 镜像源码确认）是 `(self, peft_config, adapter_name, target, target_name, parent, current_key)` ——**没有 `parameter_name`**。`inject_adapter` 内部固定用关键字调用 `current_key=key`，不传别的。所以：

- `current_key` 用关键字接收即可，不用防御。
- `**optional_kwargs` 加不加都行（不会有额外参数传进来），加上是为了未来 peft 升级增加参数时不炸，纯防御性。
- **不要**在这里处理 `inference_mode`——0.17.1 的 `inject_adapter` 会在**自己内部**统一处理 `inference_mode`（见下面 `inject_adapter` 源码摘录），不需要 `_create_and_replace` 自己 `getattr(peft_config, 'inference_mode', False)` 手动下发到 `update_layer`。这是 markdown 计划里那条 mitigation 的核心问题——它把 `inference_mode` 一路往下传到 `DenseLayer.update_layer()` → `self.set_adapter(..., inference_mode=...)`，而 `BaseTunerLayer.set_adapter` 在 0.17.1 里签名就是 `(self, adapter_names)`，**没有这个参数**，这正是 "inference_mode" TypeError 最可能的真实来源。

```python
def _create_and_replace(
    self, dense_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs,
):
    if current_key is None:
        raise ValueError("Current key cannot be None.")

    if isinstance(target, DenseLayer):
        target.update_layer(adapter_name)   # 不传 inference_mode
        return

    new_module = self._create_new_module(adapter_name, target)
    if adapter_name not in self.active_adapters:
        new_module.requires_grad_(False)
    self._replace_module(parent, target_name, new_module, target)   # 见下方"必需方法 2"
```

### 4. `_mark_only_adapters_as_trainable(self, model) -> None`

无版本坑，就是两遍 `named_parameters()`：

```python
def _mark_only_adapters_as_trainable(self, model):
    for n, p in model.named_parameters():
        if "dense_delta" not in n:
            p.requires_grad = False
    for n, p in model.named_parameters():
        if "dense_delta" in n:
            p.requires_grad = True
```

### 5 & 6. `disable_adapter_layers(self)` / `enable_adapter_layers(self)`

**这里是最容易踩的坑**：`BaseTunerLayer.disable_adapters` 是**只读 property**（源码原文：`# use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method`）。正确写法是调用 `module.enable_adapters(enabled: bool)`，**不能**直接 `module.disable_adapters = True/False`（要么报 `AttributeError: can't set attribute`，要么即使能赋值也绕过了 `requires_grad` 的正确处理）：

```python
def disable_adapter_layers(self):
    for module in self.modules():
        if isinstance(module, BaseTunerLayer):
            module.enable_adapters(enabled=False)

def enable_adapter_layers(self):
    for module in self.modules():
        if isinstance(module, BaseTunerLayer):
            module.enable_adapters(enabled=True)
```

这两个方法很可能在 KFAC 采集阶段被间接触发（`model.disable_adapter()` 上下文管理器计算参考前向时），所以哪怕暂时没报错也建议直接改对，不要等它炸。

---

## 第二类：3 个非 abstract、但实践必需的方法

这三个是 `LoraModel` 等主流 tuner **都自己写了一份**、但基类完全不强制的方法。`BaseTuner` 不提供默认实现，也不会在类定义时报错提醒你——只会在具体调用点炸 `AttributeError`。

### 1. `set_adapter(self, adapter_name)` —— 模型级，别和层级的搞混

`PeftModel.set_adapter()` 会转发到 `self.base_model.set_adapter(...)`（`self.base_model` 就是你的 `DenseModel` 实例）。更关键的是：**`BaseTuner.inject_adapter` 自己内部就会调用 `self.set_adapter(self.active_adapters)`**（0.17.1 源码原文注释：`# It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is added...`）——也就是说，`DenseModel` 一旦被构造（`get_dense_model()` 调用瞬间）就会立刻触发这个方法，不需要等到训练或推理阶段。这就是 `'DenseModel' object has no attribute 'set_adapter'` 报错的根源。

对照 `LoraModel.set_adapter`（v0.17.0 实测源码）逐行搬：

```python
def set_adapter(self, adapter_name):
    for module in self.model.modules():
        if isinstance(module, DenseLayer):
            if module.merged:
                warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                module.unmerge()
            module.set_adapter(adapter_name)   # 这里调的是层级 set_adapter，本来就有效，别改
    self.active_adapter = adapter_name
```

**不要**加 `inference_mode` 参数——`BaseTunerLayer.set_adapter` 在 0.17.1 就是 `(self, adapter_names: str | list[str])`，没有这个 kwarg，加了反而会在这里再炸一次 TypeError。

### 2. `_replace_module(self, parent, child_name, new_module, child)` —— 同样不是继承来的

`LoraModel._replace_module` 是它自己定义的方法，不是 `tuners_utils` 里能 `import` 的模块级函数。`BaseTuner._unload_and_optionally_merge`（`unload()`/`merge_and_unload()` 内部用）会调用 `self._replace_module(...)`，所以哪怕你在 `_create_and_replace` 里手写了替换逻辑，只要没有这个同名方法，未来任何调用 `.unload()` 的路径（比如调试脚本、或者 curvature_lora.py 共用逻辑里可能有的 unload 调用）都会炸：

```python
def _replace_module(self, parent, child_name, new_module, child):
    setattr(parent, child_name, new_module)

    if hasattr(child, "base_layer"):
        child = child.base_layer   # 拆开被包装的 base_layer

    weight = child.weight if hasattr(child, "weight") else next(child.parameters())

    meta = torch.device("meta")
    if not any(p.device == meta for p in new_module.parameters()):
        new_module.to(weight.device)
```

（Dense 不支持量化 base layer，所以不需要 `LoraModel._replace_module` 里处理 `qweight`/bnb 相关的分支，上面这版已经是精简过、和你现有 `_create_and_replace` 手写逻辑等价的版本。）

### 3. `__getattr__(self, name)` —— 转发给被包装的真实模型

`PeftModelForCausalLM.__init__` 会执行 `self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation`，这行代码在 `get_dense_model()` 构造阶段就会立刻跑。`self.base_model` 是 `DenseModel`，它自己没有 `prepare_inputs_for_generation`，需要透传给 `self.model`（真正被包装的 `LlamaForCausalLM`/`Qwen...ForCausalLM` 实例）。不加这个方法，任何 `DenseModel` 自己没定义、又不在 `nn.Module._parameters/_buffers/_modules` 里的属性访问（`generate`、`config`、`prepare_inputs_for_generation`……）都会直接报 `AttributeError: 'DenseModel' object has no attribute 'xxx'`：

```python
def __getattr__(self, name):
    try:
        return super().__getattr__(name)  # 先走 nn.Module 自己的查找逻辑
    except AttributeError:
        if name == "model":
            raise   # 防止 self.model 还没赋值时无限递归
        return getattr(self.model, name)
```

---

## 汇总：给 `DenseModel(BaseTuner)` 类体检的 9 项清单

| # | 方法 | 是否 abstractmethod | 不实现的后果 |
|---|---|---|---|
| 1 | `_prepare_adapter_config` | ✅ | 类定义报错 |
| 2 | `_check_target_module_exists` | ✅ | 类定义报错 |
| 3 | `_create_and_replace` | ✅ | 类定义报错 |
| 4 | `_mark_only_adapters_as_trainable` | ✅ | 类定义报错 |
| 5 | `disable_adapter_layers` | ✅ | 类定义报错（写错内容不报，但运行时 AttributeError/梯度不冻结） |
| 6 | `enable_adapter_layers` | ✅ | 同上 |
| 7 | `set_adapter` | ❌ | **构造阶段**就炸（`inject_adapter` 内部调用） |
| 8 | `_replace_module` | ❌ | `_create_and_replace` 调用处炸，或未来 `unload()` 路径炸 |
| 9 | `__getattr__` | ❌ | **构造阶段**就炸（`PeftModelForCausalLM.__init__` 里 `prepare_inputs_for_generation` 透传） |

7、8、9 三个是最危险的，因为 Python 不会在类定义时提醒你漏了，只会在具体调用点冷不丁报 `AttributeError`，而且 7 和 9 都是在**模型构造那一行**就会触发，不是训练/推理时才暴露——这也是为什么你之前是"改一个报下一个"，而不是一次性看到所有缺失：Python 的 duck typing 决定了这类缺失只能一个个撞出来，没法静态检查。按上面这份清单一次补全，应该不会再有新的"漏方法"类报错了（后续如果还有报错，大概率是逻辑 bug 而不是接口缺失了）。