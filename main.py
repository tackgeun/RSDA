import argparse
from solvers import train_init
from solvers import train
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import copy
import torch
import os

# irm solvers
from solvers import train_init_irm
from solvers import train_irm_logit, train_irm_feat, train_MSTN_irm_feat

def main(args):
    args.log_file.write('\n\n###########  initialization ############')
    
    #initializing
    if( 'default' in args.init_method):
        acc, model = train_init(args)
    elif('irm' in args.init_method):
        acc, model = train_init_irm(args)
    else:
        assert(False)
    
    init_acc = acc
    best_acc = acc
    best_model = copy.deepcopy(model)
    
    for stage in range(args.stages):
        print('\n\n########### stage : {:d}th ##############\n\n'.format(stage))
        args.log_file.write('\n\n########### stage : {:d}th    ##############'.format(stage))

        if(args.lr_decay):
            lr_conf = 'lr_decay{}_b{}'.format(args.lr, args.batch_size)
        else:
            lr_conf = 'lr{}_b{}'.format(args.lr, args.batch_size)

        if(args.lr_decay_refine):
            lr_conf_refine = 'lr_decay{}_b{}'.format(args.lr_refine, args.batch_size_refine)
        else:
            lr_conf_refine = 'lr{}_b{}'.format(args.lr_refine, args.batch_size_refine)

        # overall param
        if(args.trainable_radius):
            param1 = f'list_{args.baseline}_trainR{args.radius}'
        else:
            param1 = f'list_{args.baseline}_R{args.radius}'
    
        if(args.radius_refine > 0):
            param1 += f'_R2{args.radius_refine}'
        else:
            args.radius_refine = args.radius

        # stage 1 param
        param2 = f'_{args.init_method}_{lr_conf}'
        # stage 2 param
        param3 = f'_{args.refine_method}_{lr_conf_refine}'

        if(args.irm_weight > 0.0 or args.irm_weight_refine > 0.0):
            param1 += f'_irm{args.irm_feature}'
        
        if(args.irm_weight > 0.0):
            param2 += f'_irm{args.irm_weight}'

        if(args.irm_weight_refine > 0.0):
            param3 += f'_irm{args.irm_weight_refine}'

        if(args.stages > 0):
            args.save_path = 'data/{}/pseudo_list/{}_s{}_{}.txt'.format(args.dataset, param1 + param2, args.stages, param3)
        else:
            args.save_path = 'data/{}/pseudo_list/{}.txt'.format(args.dataset, param1 + param2)
        #updating parameters of gaussian-uniform mixture model with fixed network parameters，the updated pseudo labels and 
        #posterior probability of correct labeling is listed in folder "./data/office(dataset name)/pseudo_list"
        make_weighted_pseudo_list(args, model)
        
        #updating network parameters with fixed gussian-uniform mixture model and pseudo labels
        if(args.refine_method == 'default'):
            acc,model = train(args)
        elif('irm' in args.refine_method):
            if(args.irm_feature == 'logit'):
                acc,model = train_irm_logit(args)
            elif(args.irm_feature == 'last_hidden'): # source classification + robust pseudo label loss
                if('MSTN' in args.refine_method): # + MSTN loss
                    acc,model = train_MSTN_irm_feat(args)            
                else:
                    acc,model = train_irm_feat(args)
        else:
            assert(False)
        
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            
    torch.save(best_model,'snapshot/save/final_best_model.pk')
    print('final_best_acc:{:.4f} init_acc:{:.4f}'.format(best_acc, init_acc))
    #if(best_acc > 0.0):
    with open('%s.log' % args.perf_log, 'a+') as f:
        if(args.trainable_radius):
            pdb.set_trace()
            radius = float(best_model['radius'])
        else:
            radius = args.radius
        log1 = f"{best_acc}\t{init_acc}\t{args.stages}\t{args.irm_feature}\t"
        log2 = f"{args.init_method}\t{args.radius}\t{args.irm_weight}\t{args.lr}\t{args.lr_decay}\t{args.batch_size}\t"
        log3 = f"{args.refine_method}\t{args.radius_refine}\t{args.irm_weight_refine}\t{args.lr_refine}\t{args.lr_decay_refine}\t{args.batch_size_refine}\n"
        f.write(log1 + log2 + log3)

    return best_acc, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # parser.add_argument('--dataset',type=str,default='office')
    # parser.add_argument('--source', type=str, default='amazon')
    # parser.add_argument('--target',type=str,default='dslr')
    # parser.add_argument('--source_list', type=str, default='data/office/amazon_list.txt', help="The source dataset path list")
    # parser.add_argument('--target_list', type=str, default='data/office/dslr_list.txt', help="The target dataset path list")
    # parser.add_argument('--num_class',type=int,default=31,help='the number of classes')

    ## visda
    parser.add_argument('--dataset',type=str,default='visda-2017')
    parser.add_argument('--source', type=str, default='train')
    parser.add_argument('--target',type=str,default='validation')
    parser.add_argument('--source_list', type=str, default='data/visda-2017/train_list.txt', help="The source dataset path list")
    parser.add_argument('--target_list', type=str, default='data/visda-2017/validation_list.txt', help="The target dataset path list")    
    parser.add_argument('--num_class',type=int,default=12,help='the number of classes')
    
    # experiments
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--perf_log', type=str, default='visda-2017')
    parser.add_argument('--log_file')
 
    # parameters for overall training
    parser.add_argument('--irm_feature', type=str, default='logit')
    parser.add_argument('--irm_type', type=str, default='batch')
    parser.add_argument('--irm_warmup_step', type=int, default=10)
    parser.add_argument('--stages',type=int,default=6,help='the number of alternative iteration stages')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--radius', type=float, default=10.0)
    parser.add_argument('--trainable_radius', type=bool, default=False)

    # parameters for training stage 1
    parser.add_argument('--init_method', type=str, default='default')
    parser.add_argument('--irm_weight', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--batch_size',type=int,default=36)

    # parameters for training stage 2
    parser.add_argument('--refine_method', type=str, default='default')
    parser.add_argument('--irm_weight_refine', type=float, default=0)
    parser.add_argument('--lr_refine', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay_refine', type=bool, default=True)
    parser.add_argument('--batch_size_refine',type=int,default=36)
    parser.add_argument('--radius_refine', type=float, default=0.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not os.path.exists('snapshot'):
        os.mkdir('snapshot')
    if not os.path.exists('snapshot/{}'.format(args.output_dir)):
        os.mkdir('snapshot/{}'.format(args.output_dir))
    log_file = open('snapshot/{}/log.txt'.format(args.output_dir),'w')
    log_file.write('dataset:{}\tsource:{}\ttarget:{}\n\n'
                   ''.format(args.dataset,args.source,args.target))
    args.log_file = log_file

    main(args)




