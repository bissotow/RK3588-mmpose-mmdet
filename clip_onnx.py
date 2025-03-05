import onnx
from onnx import helper, TensorProto

# 加载 ONNX 模型
model = onnx.load('YOUR OWN MODEL PATH')


# 找到 /mlp/mlp.0/ReduceL2 节点的位置
reduce_l2_node = None
reduce_l2_index = -1
next_node = None
next_node_index = -1

# 遍历图中的所有节点，查找 ReduceL2 节点以及其下一个节点
for i, node in enumerate(model.graph.node):
    if node.name == '/mlp/mlp.0/ReduceL2':
        reduce_l2_node = node
        reduce_l2_index = i  # 记录 ReduceL2 节点的位置
        # 查找下一个节点（在模型中紧随其后的节点）
        if i + 2 < len(model.graph.node):
            next_node = model.graph.node[i + 2]
            next_node_index = i + 2
        break

# 如果没有找到 ReduceL2 节点，抛出错误
if reduce_l2_node is None:
    raise ValueError("Cannot find '/mlp/mlp.0/ReduceL2' node.")

# 创建 Constant 节点，表示常量 2.0
constant_node = helper.make_node(
    'Constant',
    inputs=[],  # Constant 不需要输入
    outputs=['const_2_output'],  # 常量输出名称
    value=helper.make_tensor(
        name='const_2',  # Tensor 名称
        data_type=TensorProto.FLOAT,  # 数据类型
        dims=[],  # 标量值，无需维度
        vals=[2.0]  # 常量值 2.0
    )
)

# 创建 Constant 节点，表示 float16 的 min 和 max 值
min_value_node = helper.make_node(
    'Constant',
    inputs=[],  # Constant 不需要输入
    outputs=['min_value_output'],  # 常量输出名称
    value=helper.make_tensor(
        name='min_value',  # Tensor 名称
        data_type=TensorProto.FLOAT16,  # 数据类型
        dims=[],  # 标量值，无需维度
        vals=[-65504.0]  # float16 最小值
    )
)

max_value_node = helper.make_node(
    'Constant',
    inputs=[],  # Constant 不需要输入
    outputs=['max_value_output'],  # 常量输出名称
    value=helper.make_tensor(
        name='max_value',  # Tensor 名称
        data_type=TensorProto.FLOAT16,  # 数据类型
        dims=[],  # 标量值，无需维度
        vals=[65504.0]  # float16 最大值
    )
)

# 创建 Clip 节点，限制输入到 [-65504, 65504]
clip_node = helper.make_node(
    'Clip',
    inputs=['pow_output', 'min_value_output', 'max_value_output'],  # 输入是 pow_output 和常量的 min/max
    outputs=['clip_output']
)

# 创建 Pow 节点，用于每个元素平方
pow_node = helper.make_node(
    'Pow',
    inputs=[reduce_l2_node.input[0], 'const_2_output'],  # 使用 constant 节点的输出作为第二个输入
    outputs=['pow_output']
)

# 创建 ReduceSum 节点，计算平方后的和
reduce_sum_node = helper.make_node(
    'ReduceSum',
    inputs=['clip_output'],  # 对 clip_output 求和
    outputs=['sum_output'],
    axes=[-1],  # 如果是按行求和，根据需要调整轴
)

# 创建 Sqrt 节点，计算平方根
sqrt_node = helper.make_node(
    'Sqrt',
    inputs=['sum_output'],  # 对求和结果取平方根
    outputs=['l2_output']
)

# 更新原始 ReduceL2 节点的输出为 l2_output
reduce_l2_node.output[0] = 'l2_output'

# 删除 ReduceL2 节点
del model.graph.node[reduce_l2_index]

# 将 Constant, Pow, Clip, ReduceSum, 和 Sqrt 节点插入到模型中
model.graph.node.insert(reduce_l2_index, constant_node)  # 在 ReduceL2 前插入 Constant 节点
model.graph.node.insert(reduce_l2_index + 1, pow_node)  # 插入 Pow 节点
model.graph.node.insert(reduce_l2_index + 2, min_value_node)  # 插入 min_value 节点
model.graph.node.insert(reduce_l2_index + 3, max_value_node)  # 插入 max_value 节点
model.graph.node.insert(reduce_l2_index + 4, clip_node)  # 插入 Clip 节点
model.graph.node.insert(reduce_l2_index + 5, reduce_sum_node)  # 插入 ReduceSum 节点
model.graph.node.insert(reduce_l2_index + 6, sqrt_node)  # 插入 Sqrt 节点

# 更新连接：确保新的节点输出连接到下一个节点的输入
if next_node:
    next_node.input[0] = 'l2_output'  # 将下一个节点的输入连接到 'l2_output'

# 验证模型
onnx.checker.check_model(model)

# 保存修改后的模型
onnx.save(model, 'YOUR OWN MODEL PATH')

print("ReduceL2 node replaced with equivalent operations successfully.")
