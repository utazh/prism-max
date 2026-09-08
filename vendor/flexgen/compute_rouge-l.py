from rouge import Rouge
import torch, sys
path1 = '/home/zrd/impllm/h2o_flexgen/flexgen/fewshots_datasets/output/sys_prompt-opt-6.7b-generate.pt'
path2 = '/home/zrd/impllm/h2o_flexgen/flexgen/fewshots_datasets/output/sys_prompt-opt-6.7b-generate-[50].pt'

data1 = torch.load(path1)
data2 = torch.load(path2)
print(len(data1[1]), len(data2[3]))
rouge = Rouge()
f1_total = 0
for result1, result2 in zip(data1, data2):
    scores = rouge.get_scores(result1, result2)
    
    f1 = scores[0]['rouge-l']['f']
    f1_total+=f1
print(f1_total/len(data1))