import argparse
import torch
import pandas as pd
import os
from sampling import sampling, greedy
from inout import load_data
from time import time
from utils import ObjMeter
import sys

# Training device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               beta: int = 32,
               seed: int = None):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: JSP instance.
        beta: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time()
    if beta > 1:
        s, mss = sampling(ins, encoder, decoder, bs=beta, device=dev)
    else:
        s, mss = greedy(ins, encoder, decoder, device=dev)
    exe_time = time() - st

    #
    _gaps = (mss / ins['makespan'] - 1) * 100
    min_gap = _gaps.min().item()
    print(f'\t- {ins["name"]} = {min_gap:.3f}%')
    results = {'NAME': ins['name'],
               'UB': ins['makespan'],
               'MS': mss.min().item(),
               'MS-AVG': mss.mean().item(),
               'MS-STD': mss.std().item(),
               'GAP': min_gap,
               'GAP-AVG': _gaps.mean().item(),
               'GAP-STD': _gaps.std().item(),
               'TIME': exe_time}
    return results


#
parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default=os.path.join("checkpoints", "PtrNet-B256.pt"),
                    help="Path to the model.")
parser.add_argument("-benchmark", type=str, required=False,
                    default='TA', help="Name of the benchmark for testing.")
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    from PointerNet import GATEncoder
    print(f"Using {dev}...")

    # Load the model
    print(f"Loading {args.model_path}")
    try:
        # Try loading old format first
        enc_w, dec_ = torch.load(args.model_path, map_location=dev)
        enc_ = GATEncoder(15).to(dev)   # Load weights to avoid bug with new PyG
        enc_.load_state_dict(enc_w)
    except Exception as e:
        print(f"Error loading old format model: {e}")
        print("Trying new format...")
        try:
            # Try loading new format
            weights = torch.load(args.model_path, map_location=dev)
            if isinstance(weights, dict) and 'encoder_state_dict' in weights and 'decoder_state_dict' in weights:
                enc_ = GATEncoder(15).to(dev)
                enc_.load_state_dict(weights['encoder_state_dict'])
                from PointerNet import MHADecoder
                dec_ = MHADecoder(encoder_size=enc_.out_size,
                                 context_size=11,  # JobShopStates.size
                                 hidden_size=64,
                                 mem_size=128,
                                 clf_size=64).to(dev)
                dec_.load_state_dict(weights['decoder_state_dict'])
                print("Successfully loaded model using new format")
            else:
                print(f"Unexpected model format: {type(weights)}")
                sys.exit(1)
        except Exception as e2:
            print(f"Error loading new format model: {e2}")
            sys.exit(1)
            
    m_name = os.path.basename(args.model_path).split('.', 1)[0]

    #
    benchmark_path = os.path.join('benchmarks', args.benchmark)
    if not os.path.exists('output'):
        os.makedirs('output')
    out_file = os.path.join('output', f'{m_name}_{args.benchmark}-B{args.beta}_{args.seed}.csv')

    #
    gaps = ObjMeter()
    for file in os.listdir(benchmark_path):
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data(os.path.join(benchmark_path, file), device=dev)
        res = validation(enc_, dec_, instance,
                         beta=args.beta, seed=args.seed)

        # Save results
        pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',')
        gaps.update(instance, res['GAP'])

    #
    print(f"\t\t{args.benchmark} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
