"""Verify actual HTTP outputs and operator execution, not just loading a module."""
import argparse
import json
import requests
from common import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8003');args=parser.parse_args()
    session=requests.Session();session.trust_env=False
    health=session.get(args.url+'/health',timeout=10);health.raise_for_status()
    results=[]
    try:
        for prompt in ['用一句话解释什么是矩阵。','梯度下降中学习率有什么作用？','公平硬币抛两次，两次正面的概率是多少？']:
            outputs={}
            for backend in ['baseline','fused']:
                response=session.post(args.url+'/generate',json={'prompt':prompt,'backend':backend,'max_new_tokens':32},timeout=(5,180))
                response.raise_for_status();outputs[backend]=response.json()
            assert outputs['baseline']['ids']==outputs['fused']['ids']
            assert outputs['fused']['patched_modules']==57 and outputs['fused']['fallbacks']==0
            assert outputs['fused']['custom_kernel_calls']==57*outputs['fused']['output_tokens']
            results.append({'prompt':prompt,'same_ids':True,'outputs':outputs})
        invalid=session.post(args.url+'/generate',json={'prompt':'test','backend':'bad'},timeout=10)
        assert invalid.status_code==422
        save('35_http.json',{'health':health.json(),'cases':results,'invalid_backend_422':True,
                            'scope':'Actual deployed custom kernels, EOS-aware greedy, same IDs on three prompts; HTTP timings are observations, not a controlled benchmark.'})
        print(json.dumps({'same_ids_on_prompts':len(results),'real_kernel_calls_verified':True},indent=2))
    finally:
        session.close()


if __name__=='__main__':main()
