"""CPU-only arithmetic: instruction indexing, traffic budgets, Amdahl's law."""
from common import save


def main():
    n,threads=1003,256
    blocks=(n+threads-1)//threads
    valid=[b*threads+t for b in range(blocks) for t in range(threads) if b*threads+t<n]
    assert valid==list(range(n))
    fraction,speed=0.12,4
    overall=1/((1-fraction)+fraction/speed)
    save('28_cost_model.json',{'n':n,'threads':threads,'blocks':blocks,'launched_threads':blocks*threads,
         'masked_threads':blocks*threads-n,'rms_width':1536,'fp16_row_minimum_data_bytes':1536*2*3,
         'amdahl':{'original_fraction':fraction,'operator_speedup':speed,'overall_speedup':overall},
         'scope':'Arithmetic model, not GPU measurement or proof of model speedup.'})
    print(f'blocks={blocks}; masked={blocks*threads-n}; Amdahl={overall:.4f}x')


if __name__=='__main__':
    main()
