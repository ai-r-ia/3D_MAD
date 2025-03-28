import os
from multiprocessing import Pool
import subprocess
from configs.config import create_parser
from configs.seed import set_seed

def main(args):
    cmds = [
        f'python train/lbp_svm.py --trainds {args.trainds} --testds {args.testds}',     
        f'python train/vit_svm.py --trainds {args.trainds} --testds {args.testds}',     
        f'python train/resnet_svm.py --trainds {args.trainds} --testds {args.testds}',     
        f'python point_clouds/train_pc.py --trainds {args.trainds} --testds {args.testds} --pc_model pointnet',     
        f'python point_clouds/train_pc.py --trainds {args.trainds} --testds {args.testds} --pc_model pointnet2',     
        f'python point_clouds/train_pc.py --trainds {args.trainds} --testds {args.testds} --pc_model pointnet2_simpleview'
    ]
    
    processes = [subprocess.Popen(cmd, shell=True) for cmd in cmds]
    
    for p in processes:
        p.wait()  
                
if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args)

