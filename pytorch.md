# PyTorch Tricks

### TorchModel模型保存

torch只保存模型实例的成员变量，且成员变量必须为torch类型的变量，不能保存非torch类型的变量（如使用list包起来的parameters需要使用ModuleList包起来）

```python
model.state_dict()
```

