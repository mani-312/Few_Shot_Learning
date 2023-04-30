import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]
    best_model_path = args.save_path + "/best_model.pth"
    last_model_path = args.save_path + "/last_model.pth"
    #device = torch.device('cuda:2')
    print("Device == ",args.device)
    maml = Meta(args, config).to(args.device)

    if(args.model_path != ''):
        print("Loading Pre-trained model at path == ",args.model_path)
        model_dict = torch.load(args.model_path)
        maml.load_state_dict(model_dict)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/data4/home/manikantab/virtual_env/MAML-Pytorch', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('/data4/home/manikantab/virtual_env/MAML-Pytorch', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    #mini_val = MiniImagenet('/data4/home/manikantab/virtual_env/MAML-Pytorch', mode='val', n_way=args.n_way, k_shot=args.k_spt,
    #                         k_query=args.k_qry,
    #                         batchsz=100, resize=args.imgsz)
    
    #print("Train == ",mini.shape)
    #print("Test == ",mini_test.shape)
    best_acc = 0.0

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            if step == 0:
                print("Step == ",step)
                print("SUpport_X == ",x_spt.shape) #meta_batch_size * #images_in_support_set * image_size
                print("Support_y == ",y_spt.shape)
                print("QUery_X == ",x_qry.shape)
                print("Query_y == ",y_qry.shape)

            x_spt, y_spt, x_qry, y_qry = x_spt.to(args.device), y_spt.to(args.device), x_qry.to(args.device), y_qry.to(args.device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # validation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_val = []

                for t_step,(x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
                    #if(t_step%30 == 0):
                    #    print("----------- Val_step == ",t_step)
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                                                 x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_val.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_val).mean(axis=0).astype(np.float16)
                print('Val acc:', accs)
                if(np.mean(accs[-5:]) > best_acc):
                    print(" ----------- Saving the best model ------- \n\n")
                    torch.save(maml.state_dict(), best_model_path)
                    best_acc = accs.mean()
    
       
    torch.save(maml.state_dict(), last_model_path)
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                                                 x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)

        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_path",type = str,help = "Model saving path",default='/data4/home/manikantab/virtual_env/MAML-Pytorch/models')
    argparser.add_argument("--model_path",type = str,help = "Pretrained_model_path",default='')
    argparser.add_argument('--device',type = str,help = 'Computation device',default = "cuda:0")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
