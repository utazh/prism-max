import argparse
import json
import os, sys
from datetime import datetime

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor

def json_to_key(obj):
    return json.dumps(obj)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_filename(filepath, exp, args):
    """参数 exp 由命令行传入"""
    dirname, file_ext = os.path.dirname(filepath), os.path.basename(filepath)
    filen, ext = os.path.splitext(file_ext)
    if args.samekv:
        outputfilen = filen + f'-samekv-eval-result-{args.sele_percent}-{exp}.log'
    elif args.motiv2:
        outputfilen = filen + f'-motiv2-eval-result-{args.sele_percent}-{exp}.log'
    else:
        outputfilen = filen + f'-eval-result-{exp}.log'
    return os.path.join(dirname, outputfilen)

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='result.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--sele-percent', type=str, default='100')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--exp', '-e', type=str, required=True, help="denotes for log file's name")
    parser.add_argument("--motiv2", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--samekv", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    
    # 保存原始的 sys.stdout 和 sys.stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # 把输出到终端的内容重定向到 logfile
    logfile = get_filename(args.result_file, args.exp, args)
    with open(logfile, 'w') as out_f:
        # 将 sys.stdout 和 sys.stderr 重定向到文件
        sys.stdout = out_f
        sys.stderr = out_f
        
        print(f'-> log time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        try:
            if args.model_type == 'opt':
                os.environ['MODEL_NAME'] = "facebook/opt-66b"
            elif args.model_type == 'bloom':
                os.environ['MODEL_NAME'] = "bigscience/bloom"
            elif args.model_type == 'gpt_neox':
                os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
            elif args.model_type == 'llama':
                os.environ['MODEL_NAME'] = "huggyllama/llama-7b"
            else:
                assert False

            seq = 1024
            total_batch = 1
            pe = 'fixed'

            class RealRunner:
                
                def __init__(self, args):
                    
                    self.results = {}
                    
                    with open(args.result_file, 'r') as f:
                        
                        for line in f:
                            if line.strip() == '':
                                continue
                            
                            item = json.loads(line)
                            
                            request = item['request']
                            result = item['result']
                            
                            self.results[json_to_key(request)] = result
                    
                    print(f"{len(self.results)} items in the cache")
                
                def eval(self, batch):
                    
                    from tasks.eval_harness import tokenizer
                    
                    mask_loss = []
                    each_correct = []

                    for i, text in enumerate(batch['text']):
                        
                        request = {
                                "best_of": 1, 
                                "echo": True, 
                                "logprobs": 1, 
                                "max_tokens": 0, 
                                "model": "x", 
                                "n": 1, 
                                "prompt": text, 
                                "request_type": "language-model-inference", 
                                "stop": None, 
                                "temperature": 0, 
                                "top_p": 1
                            }
                        
                        key = json_to_key(request)
                        
                        correct = True
                        
                        if key in self.results:
                            result = self.results[key]
                            
                            token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                            tokens = result['choices'][0]['logprobs']['tokens']
                            top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                            assert token_logprobs[0] is None
                            
                            token_ids = tokenizer.convert_tokens_to_ids(tokens)
                            
                            obs = batch['obs'][i]
                            target = batch['target'][i]
                            eval_mask = batch['eval_mask'][i]
                            
                            n_positive = 0
                            sum_lobprob = 0
                            if args.debug:
                                print(target)
                            for i, mask in enumerate(eval_mask):
                                try:
                                    
                                    if i+1 >= len(tokens):
                                        break
                                    
                                    if mask == True:
                                        if args.debug:
                                            print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                        correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                        sum_lobprob += token_logprobs[i+1]
                                        n_positive += 1
                                except Exception as e:
                                    raise e
                            
                            # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                            avg_logprob = sum_lobprob / n_positive
                            
                            mask_loss.append( - avg_logprob)
                    
                            each_correct.append( correct )
                            
                        else:
                            assert False
                        

                    out = {
                        'mask_loss': mask_loss,
                        'each_correct': each_correct,
                    }
                    
                    
                    return out

            t = RealRunner(args)

            adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

            results = evaluator.evaluate(adaptor, tasks.get_task_dict([args.task_name
                                                                    #"lambada_openai",
                                                                    #"piqa",
                                                                    #"hellaswag",
                                                                    #"winogrande",
                                                                    #"mathqa",
                                                                    #"pubmedqa",
                                                                    # "boolq",
                                                                    # "cb",
                                                                    # "copa",
                                                                    # "multirc",
                                                                    # "record",
                                                                    # "wic",
                                                                    # "wsc",
                                                                    ]), False)
            
            dumped = json.dumps(results, indent=2)
            print(dumped)
        finally:
            # 恢复原始的 sys.stdout 和 sys.stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
