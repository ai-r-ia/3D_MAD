import os
from multiprocessing import Pool
from configs.config import create_parser
from configs.seed import set_seed

def main(args):
    cmds = []
    
    if args.modeltype == 'sota':
        # args.append(
        #     f"python train_sota.py --trainds {args.trainds} --testds {args.testds} -rdir {args.rdir}"
        # )
        cmds.append(
            f"python train/train_sota.py --trainds iPhone11 --testds iPhone11"
        )
        cmds.append(
            f"python train/train_sota.py --trainds iPhone12 --testds iPhone12"
        )
        cmds.append(
            f"python train/train_sota.py --trainds iPhone11 --testds iPhone12"
        )
        cmds.append(
            f"python train/train_sota.py --trainds iPhone12 --testds iPhone11"
        )
    elif args.modeltype =='proposed':

        if args.runtype == 'train':
            # args.append(
            #     f"python train.py --trainds {args.trainds} --testds {args.testds} -rdir {args.rdir}"
            # )
            # cmds.append(
            #     f"python train/train.py --trainds iPhone11"
            # )
            cmds.append(
                f"python train/train.py --trainds iPhone12"
            )
            
        elif args.runtype == 'test':
            cmds.append(
                f"python train/test.py --trainds iPhone11 --testds iPhone11"
            )
            # cmds.append(
            #     f"python train/test.py --trainds iPhone12 --testds iPhone12"
            # )
            cmds.append(
                f"python train/test.py --trainds iPhone11 --testds iPhone12"
            )
            # cmds.append(
            #     f"python train/test.py --trainds iPhone12 --testds iPhone11"
            # )            
    
    with Pool(1) as pool:
        pool.map(os.system, cmds)
    
if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
   
   
# to run this file: (additionally, add dataset names and root directory if required)
# python main.py -model sota 
# python  main.py -model proposed -run train
# python  main.py -model proposed -run test