import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, required=True, help="Rank of the node (0, 1, 2, or 3)")
    parser.add_argument("--world_size", type=int, default=4, help="Total number of nodes")
    parser.add_argument("--master_ip", type=str, default="10.10.1.4", help="IP of Node 0")
    parser.add_argument("--master_port", type=str, default="12345", help="Port for communication")
    args = parser.parse_args()

    print(f"Node {args.local_rank} attempting to connect to {args.master_ip}:{args.master_port}...")

    # 1. Initialize process group (Exactly as requested by the FAQ)
    dist.init_process_group(
        backend='gloo', 
        init_method=f"tcp://{args.master_ip}:{args.master_port}", 
        world_size=args.world_size, 
        rank=args.local_rank
    )
    
    print(f"SUCCESS! Node {args.local_rank} successfully connected to the group.")

    # 2. Test Communication: Let's do an All-Reduce
    # Every node creates a tensor with the value 1.0
    tensor = torch.tensor([1.0])
    
    # Wait for all nodes to reach this point
    dist.barrier()
    
    # Sum the tensors across all nodes. 
    # If 4 nodes send "1.0", the result should be "4.0" on every node.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Node {args.local_rank} final tensor value: {tensor.item()} (Should be 4.0)")

if __name__ == "__main__":
    main()
