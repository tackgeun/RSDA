import argparse
from solvers import train_init, train_init_irm
from solvers import train,train_irm,train_irm_feat,train_MSTN_irm_feat
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import copy
import torch
import os

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

        if(args.irm_weight > 0.0):
            args.save_path = 'data/{}/pseudo_list/{}_{}_list_irm-{}_{}.txt'.format(args.dataset,args.source,args.target,args.irm_feature,args.irm_weight)
        else:
            args.save_path = 'data/{}/pseudo_list/{}_{}_list.txt'.format(args.dataset,args.source,args.target)        

        #updating parameters of gaussian-uniform mixture model with fixed network parameters，the updated pseudo labels and 
        #posterior probability of correct labeling is listed in folder "./data/office(dataset name)/pseudo_list"
        make_weighted_pseudo_list(args, model)
        
        #updating network parameters with fixed gussian-uniform mixture model and pseudo labels
        if(args.irm_weight > 0.0):
            if(args.irm_feature == 'logit'):
                acc,model = train_irm(args)
            elif(args.irm_feature == 'last_hidden'): # source classification + robust pseudo label loss
                acc,model = train_irm_feat(args)
            elif(args.irm_feature == 'last_hidden_MSTN'): # + MSTN loss
                acc,model = train_MSTN_irm_feat(args)
        else:
            acc,model = train(args)
        
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            
    torch.save(best_model,'snapshot/save/final_best_model.pk')
    print('final_best_acc:{:.4f} init_acc:{:.4f}'.format(best_acc, init_acc))
    return best_acc,best_model


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
    
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--irm_weight', type=float, default=0.0)
    parser.add_argument('--irm_feature', type=str, default='logit')
    parser.add_argument('--irm_type', type=str, default='batch')
    parser.add_argument('--irm_warmup_step', type=int, default=10)
    parser.add_argument('--init_method', type=str, default='default')

    parser.add_argument('--stages',type=int,default=6,help='the number of alternative iteration stages')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--batch_size',type=int,default=36)
    parser.add_argument('--log_file')
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




