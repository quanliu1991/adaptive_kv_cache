# Adaptive_kv_cache
自适应压缩KV Cache Attention方法，减少大模型推理过程中KV Cache显存占用。参考ICLR 2024论文[《MODEL TELLS YOU WHAT TO DISCARD:
ADAPTIVE KV CACHE COMPRESSION FOR LLMS》](https://openreview.net/pdf?id=uNrFpDPMyo)实现。

**测试结论**：
在llama2-7B模型中，超参设置为（T=0.95、r_f=0.8，r_l=0.8）时，输入token长度为18，平均每层注意力显存由0.281MB下降到0.239MB，平均减少14.77%。  
其中：  
T=0.95，表示KV cache 压缩后attention score的恢复阈值为0.95。  
r_f=0.8，表示使用KV cache 压缩时保留前80%最高分数的token作为kv cache。  
r_l=0.8，表示使用KV cache 压缩时保留最近的输入token长度80%的token作为kv cache。  


1. 每层显存占用表

| layer_id  | raw KV cache mem MB| compress KV cache mem MB| compass ratio % |
|-----------|------------------|-----------------------|-----------------|
| total ave | 0.281              | 0.239                 | 14.77           |
| layer0    | 0.281              | 0.256                 | 9.028           |
| layer1    | 0.281              | 0.249                 | 11.458          |
| layer2    | 0.281              | 0.250                 | 11.111          |
| layer3    | 0.281              | 0.243                 | 13.542          |
| layer4    | 0.281              | 0.245                 | 13.021          |
| layer5    | 0.281              | 0.244                 | 13.368          |
| layer6    | 0.281              | 0.241                 | 14.410          |
| layer7    | 0.281              | 0.243                 | 13.542          |
| layer8    | 0.281              | 0.239                 | 14.931          |
| layer9    | 0.281              | 0.245                 | 12.847          |
| layer10   | 0.281              | 0.250                 | 11.111          |
| layer11   | 0.281              | 0.247                 | 12.153          |
| layer12   | 0.281              | 0.232                 | 17.361          |
| layer13   | 0.281              | 0.243                 | 13.542          |
| layer14   | 0.281              | 0.247                 | 12.153          |
| layer15   | 0.281              | 0.243                 | 13.542          |
| layer16   | 0.281              | 0.250                 | 11.111          |
| layer17   | 0.281              | 0.245                 | 12.847          |
| layer18   | 0.281              | 0.245                 | 13.021          |
| layer19   | 0.281              | 0.237                 | 15.625          |
| layer20   | 0.281              | 0.245                 | 12.847          |
| layer21   | 0.281              | 0.240                 | 14.583          |
| layer22   | 0.281              | 0.229                 | 18.750          |
| layer23   | 0.281              | 0.232                 | 17.535          |
| layer24   | 0.281              | 0.230                 | 18.056          |
| layer25   | 0.281              | 0.224                 | 20.486          |
| layer26   | 0.281              | 0.224                 | 20.486          |
| layer27   | 0.281              | 0.228                 | 19.097          |
| layer28   | 0.281              | 0.215                 | 23.438          |
| layer29   | 0.281              | 0.223                 | 20.66           |
| layer30   | 0.281              | 0.228                 | 18.924          |
| layer31   | 0.281             | 0.258                 | 8.160           |



# 代码结构及功能
```
├── adapter_kv_cache
│   ├── compression_policies.py  # 压缩策略
│   ├── configs
│   │   └── config.json  # 超参数配置文件
│   ├── kv_cache
│   │   ├── kv_cache_manager.py  # kv cache缓存管理
│   ├── kv_cache_adapter.py  # 自适应kv cache处理器
│   ├── layers
│   │   ├── adaptive_kv_attation.py # 自适应kv cache注意力
│   ├── models
│   └── utils.py # 通用工具
├── images # 当使用环境变量SAVE_IMAGE=1时，保存层注意力每个头的注意力热力图
├── LICENSE
├── llama # 基于https://github.com/meta-llama/llama修改，用于自适应kv cache注意力效果测试
├── logs
│   └── test.log
└── README.md
```

# 预实验


<img src="images/layer0_attention.png" width="30%">


# 代码实现


# 显存占用测试


# 推理结果测试


# 其他限制


1. 计算attention sroces相似的方法
   - 目的是尽量保证输出的o一致
   - 每层32个头需要分别计算，分别给出每个头的压缩策略
2. 
    