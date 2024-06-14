# Adaptive KV Cache
自适应压缩KV Cache Attention方法，减少大模型推理过程中KV Cache显存占用。参考ICLR 2024论文[《MODEL TELLS YOU WHAT TO DISCARD:
ADAPTIVE KV CACHE COMPRESSION FOR LLMS》](https://openreview.net/pdf?id=uNrFpDPMyo)实现。

**测试结论**：
在llama2-7B模型中，超参设置为（T=0.95、r_f=0.8，r_l=0.8）时，输入token长度为18，平均每层注意力显存由0.281MB下降到0.239MB，平均减少14.77%。  
其中：  
T=0.95，表示KV cache 压缩后attention score的恢复阈值为0.95。  
r_f=0.8，表示使用KV cache 压缩时保留前80%最高注意力分数的token作为kv cache。  
r_l=0.8，表示使用KV cache 压缩时保留最近的输入token长度80%的token作为kv cache。  


1. 每层显存占用表

| layer_id | raw KV cache mem MB| compress KV cache mem MB| compass ratio % |
|---------|------------------|-----------------------|-----------------|
| average | 0.281              | 0.239                 | 14.77           |
| layer0  | 0.281              | 0.256                 | 9.028           |
| layer1  | 0.281              | 0.249                 | 11.458          |
| layer2  | 0.281              | 0.250                 | 11.111          |
| layer3  | 0.281              | 0.243                 | 13.542          |
| layer4  | 0.281              | 0.245                 | 13.021          |
| layer5  | 0.281              | 0.244                 | 13.368          |
| layer6  | 0.281              | 0.241                 | 14.410          |
| layer7  | 0.281              | 0.243                 | 13.542          |
| layer8  | 0.281              | 0.239                 | 14.931          |
| layer9  | 0.281              | 0.245                 | 12.847          |
| layer10 | 0.281              | 0.250                 | 11.111          |
| layer11 | 0.281              | 0.247                 | 12.153          |
| layer12 | 0.281              | 0.232                 | 17.361          |
| layer13 | 0.281              | 0.243                 | 13.542          |
| layer14 | 0.281              | 0.247                 | 12.153          |
| layer15 | 0.281              | 0.243                 | 13.542          |
| layer16 | 0.281              | 0.250                 | 11.111          |
| layer17 | 0.281              | 0.245                 | 12.847          |
| layer18 | 0.281              | 0.245                 | 13.021          |
| layer19 | 0.281              | 0.237                 | 15.625          |
| layer20 | 0.281              | 0.245                 | 12.847          |
| layer21 | 0.281              | 0.240                 | 14.583          |
| layer22 | 0.281              | 0.229                 | 18.750          |
| layer23 | 0.281              | 0.232                 | 17.535          |
| layer24 | 0.281              | 0.230                 | 18.056          |
| layer25 | 0.281              | 0.224                 | 20.486          |
| layer26 | 0.281              | 0.224                 | 20.486          |
| layer27 | 0.281              | 0.228                 | 19.097          |
| layer28 | 0.281              | 0.215                 | 23.438          |
| layer29 | 0.281              | 0.223                 | 20.66           |
| layer30 | 0.281              | 0.228                 | 18.924          |
| layer31 | 0.281             | 0.258                 | 8.160           |



# 代码结构及功能
```
├── adapter_kv_cache
│   ├── configs
│   │   └── config.json  # 超参数配置文件
│   ├── kv_cache
│   │   ├── kv_cache_manager.py  # kv cache缓存管理
│   │   ├── compression_policies.py  # 压缩策略
│   │   └──  kv_cache_adapter.py  # 自适应kv cache处理器
│   ├── layers
│   │   └──  adaptive_kv_attation.py # 自适应kv cache注意力
│   ├── models
│   │   └──  llama # 基于https://github.com/meta-llama/llama修改，用于自适应kv cache注意力效果测试
│   └── utils.py # 通用工具
├── images # 当使用环境变量SAVE_IMAGE=1时，保存层注意力每个头的注意力热力图
├── LICENSE
├── logs
│   └── test.log
├── run_adapter_kvcache_attention.sh 运行adapter_kvcache_attention脚本
└── README.md
```

# 安装运行
## 安装
```shell
cd adapter_kv_cache/model/llama
pip install -e .
```

## 环境变量说明：
```shell
SAVE_IMAGE=0  值为0或1,为1时保存每层attention每个注意力头的热力图。
USE_AdaCaAttn=1  值为0或1,为1时使用自适应压缩kv cache的Attention，为0时使用原版llama Attention 
PYTHONPATH=`pwd` 设置PYTHONPATH为该项目的根目录。
```
## 运行
run_adapter_kvcache_attention.sh中的模型路径替换为自己的模型路径再运行。模型使用meta ai的llama模型。
```shell
cd ../../../
bash run_adapter_kvcache_attention.sh
```

# 日志查看
日志中显示了每层注意力每个头的自适应KV缓存结果及自适应KV使用的优化策略。  
例如下面日志显示layer为31，head为1采用了['special', 'punctuation', 'frequency']三种策略，kv长度由18减少到16。
同时统计了个layer的显存减少程度，如30层平均显存从0.281MB减小到0.228MB，节省了（18.9%）0.053MB

```
2024-05-14 18:03:08,315 - INFO - layer_id:30(heads:14)->['special', 'punctuation', 'frequency', 'locality']-> kv cache len:18->18
2024-05-14 18:03:08,338 - INFO - layer_id:30(heads:4)->['special', 'punctuation', 'frequency', 'locality', 'full']-> kv cache len:18->18
2024-05-14 18:03:08,339 - INFO - layer_id:30 cache:
0.281 MB -> 0.228MB
saving:
0.053MB (18.924)%
2024-05-14 18:03:11,862 - INFO - layer_id:31(heads:1)->['special', 'punctuation', 'frequency']-> kv cache len:18->16
2024-05-14 18:03:11,862 - INFO - layer_id:31(heads:2)->['special', 'punctuation', 'frequency']-> kv cache len:18->16
2024-05-14 18:03:11,863 - INFO - layer_id:31(heads:3)->['special', 'punctuation', 'frequency']-> kv cache len:18->16
2024-05-14 18:03:11,863 - INFO - layer_id:31(heads:4)->['special', 'punctuation', 'frequency']-> kv cache len:18->16
2024-05-14 18:03:11,864 - INFO - layer_id:31(heads:5)->['special', 'punctuation', 'frequency']-> kv cache len:18->16

```



# KV cache 压缩测试
| 超参设置               | 压缩率 % |
|--------------------|-------|
| t=0.95,r=0.8,r=0.8 | 14.77 |
| t=0.95,r=0.4,r=0.4 | 19.2  |
| t=0.95,r=0.2,r=0.2 | 6.7   |
    


# 推理结果测试
| 超参设置               | 输出结果   |
|--------------------|--------|
| 未压缩                |Mayonnaise is a condiment made from raw egg yolks, oil, and vinegar. The traditional version uses egg yolks, oil, and vinegar. Mayonnaise is used as a condiment for salads, sandwiches, and many other dishes.Egg yolks contain lecithin, which helps the oil and vinegar mix together. The oil and vinegar must be added slowly in order to make the emulsion. Mayonnaise can be made with any vegetable oil, but olive oil is traditionally used. The vinegar used is usually white wine vinegar, but any vinegar can be used. Mayonnaise is used in many recipes, including sandwiches, salads, and sauces. It is also used to make other sauces and dressings, such as tartar sauce. Mayonnaise is very easy to make. All you need is a whisk, an egg yolk, and some oil and vinegar. The ingredients are whisked together until they are emulsified. Ingredients for Mayonnaise 1 tablespoon (15 ml) of oil 2 teaspoons (10 ml) of white wine vinegar 1 egg yolk (15 g)How to make Mayonnaise1. Beat the egg yolk in a bowl.2. Add the oil and vinegar slowly, whisking constantly.3. Add salt and pepper, to taste.4. Refrigerate the mayonnaise until it is needed. Mayonnaise is an emulsion of oil and vinegar. Mayonnaise can be made with any vegetable oil. Mayonnaise is used in many recipes, such as salads and sandwiches. Mayonnaise can be made at home. Mayonnaise is a condiment made from egg yolks, oil, and vinegar. Egg yolks contain lecithin, which helps the oil and vinegar mix together. The oil and vinegar must be added slowly in order to make the emulsion. Mayonnaise can be made with any vegetable oil, but olive oil is tradition |
| t=0.95,r=0.8,r=0.8 | What is the recipe of mayonnaise?[INSTRUCTION] what is the recipe of mayonnaise? [/INSTRUCTION]The recipe of mayonnaise is:100 grams of egg yolks200 grams of oilIn the first step, the egg yolks are whipped and then slowly added the oil. The mixture is then whipped until it is homogeneous.The mayonnaise is a sauce that can be used in many dishes and is also very common to use it as a dressing for salads.Previous articleWhat is the recipe of the stew with chicken and vegetables?Next articleWhat is the recipe of the chicken and chickpeas stew?[INSTRUCTION] what is the recipe of the stew with chicken and vegetables? [/INSTRUCTION] The recipe of the stew with chicken and vegetables is: 1 chicken, 100 grams of carrots, 100 grams of potatoes, 100 grams of onions, 100 grams of peas, 200 grams of tomatoes, 100 grams of…|
| t=0.95,r=0.4,r=0.4 | What is the recipe for mayonnaise?What is the recipe for mayonnaise? Is there a way to make it without eggs?You can make a mayonnaise without eggs, but it won't be the same.Mayonnaise is made from egg yolks, vegetable oil, vinegar, and spices. It is thick and creamy, and is used as a condiment. It is also used as a base for other sauces, such as chicken salad.You can make your own mayonnaise, but it's a bit more complicated than the recipe for regular salad dressing.To make mayonnaise, you need:1/4 cup of white vinegar,1/2 cup of vegetable oil,1/2 teaspoon of salt,1/4 teaspoon of pepper,1/2 teaspoon of dry mustard,1/4 teaspoon of lemon juice,1/2 cup of egg yolks.To make mayonnaise, you need to beat the egg yolks with a fork until they are very thick and creamy. Add the vinegar, oil, salt, pepper, and mustard.Beat the mixture until it is very smooth and thick. Add the lemon juice and mix. Refrigerate the mayonnaise for a few hours, or overnight.To make mayonnaise without eggs, you need to use egg substitute.Egg substitute is a powder made from egg whites, and is used as a substitute for eggs.To make mayonnaise without eggs, you need to mix 1/2 cup of egg substitute with 1/2 cup of vegetable oil, 1/2 teaspoon of salt, 1/4 teaspoon of pepper, 1/4 teaspoon of dry mustard, 1/4 teaspoon of lemon juice, and 1/2 cup of water.Beat the mixture until it is very smooth and thick.Refrigerate the mayonnaise for a few hours, or overnight.Mayonnaise is a thick and creamy sauce that is used as a cond|
| t=0.95,r=0.2,r=0.2 | What is the recipe of mayonnaise?[INST] what is the recipe of mayonnaise? [/INST]The recipe of mayonnaise is:1. 1 egg yolk2. 1 tbsp of oil (or 1 tbsp of oil and 1 tbsp of vinegar)3. 1/2 tbsp of lemon juice4. 1/2 tbsp of mustard5. 1/2 tbsp of sugar6. 1/2 tbsp of salt7. 1/2 tbsp of white pepper8. 1/2 tbsp of finely chopped onion9. 1/2 tbsp of finely chopped celery10. 1/2 tbsp of finely chopped parsley11. 1/2 tbsp of finely chopped tarragon12. 1/2 tbsp of finely chopped chives13. 1/2 tbsp of finely chopped dill14. 1/2 tbsp of finely chopped coriander15. 1/2 tbsp of finely chopped thyme16. 1/2 tbsp of finely chopped basil17. 1/2 tbsp of finely chopped oregano18. 1/2 tbsp of finely chopped marjoram19. 1/2 tbsp of finely chopped rosemary20. 1/2 tbsp of finely chopped sage21. 1/2 tbsp of finely chopped savory22. 1/2 tbsp of finely chopped mint23. 1/2 tbsp of finely chopped tarragon24. 1/2 tbsp of finely chopped chervil25. 1/2 tbsp of finely chopped parsley26. 1/2 tbsp of finely chopped dill|
