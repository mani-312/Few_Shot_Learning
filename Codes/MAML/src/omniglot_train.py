import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
from mnist_dataloader import MNIST_dataloader
from    meta import Meta

def main(args):

    best_model_path = args.save_path + "/best_model.pth"
    last_model_path = args.save_path + "/last_model.pth"
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda:3')
    maml = Meta(args, config).to(device)

    if(args.model_path != ''):
        print("Loading Pre-trained model at path == ",args.model_path)
        model_dict = torch.load(args.model_path)
        maml.load_state_dict(model_dict)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('/data4/home/manikantab/ML_Course_project/Prototypical-Networks-for-Few-shot-Learning-PyTorch-master/dataset',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    best_acc = 0.0
    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        if step == 0:
            print("----- Train ------")
            print("x_spt.shape == ",x_spt.shape)
            print("y_spt.shape == ",y_spt.shape)
            print("x_qry.shape == ",x_qry.shape)
            print("y_qry.shape == ",y_qry.shape)
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 500 == 0:
            accs = []
            for step_t in range(1000//args.task_num):
                
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                if step_t == 0:
                    print("----- Test ------")
                    print("x_spt.shape == ",x_spt.shape)
                    print("y_spt.shape == ",y_spt.shape)
                    print("x_qry.shape == ",x_qry.shape)
                    print("y_qry.shape == ",y_qry.shape)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            if(np.mean(accs[-5:]) > best_acc):
                    print(" ----------- Saving the best model ------- \n\n")
                    torch.save(maml.state_dict(), best_model_path)
                    best_acc = np.mean(accs[-5:])

    torch.save(maml.state_dict(), last_model_path)

def eval(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]
    device = torch.device('cuda:3')
    maml = Meta(args, config).to(device)

    print("Loading Pre-trained model at path == ",args.model_path)
    model_dict = torch.load(args.model_path)
    maml.load_state_dict(model_dict)

    if(args.eval=='omniglot'):
        db_train = OmniglotNShot('/data4/home/manikantab/ML_Course_project/Prototypical-Networks-for-Few-shot-Learning-PyTorch-master/dataset',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
    else:
        db_train = MNIST_dataloader('./data',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
        
    accs = []
    for step_t in range(1000//args.task_num):
                
                # test
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        if step_t == 0:
            print("----- Test ------")
            print("x_spt.shape == ",x_spt.shape)
            print("y_spt.shape == ",y_spt.shape)
            print("x_qry.shape == ",x_qry.shape)
            print("y_qry.shape == ",y_qry.shape)

                # split to single task each time
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            # test_acc are of size args.update_step_test, i.e accuracy after each update
            accs.append( test_acc )
        if step_t % 5 == 0:
            print('step:', step_t, '\ttest acc:', np.array(accs).mean(axis=0).astype(np.float16))

    accs = np.array(accs).mean(axis=0).astype(np.float16)
    import matplotlib.pyplot as plt
    plt.plot(accs,'-o')
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.title("1-shot 5-way model with gradient updates during training = 5")
    plt.show()
    plt.savefig("plots/gradient_steps_vs_Accuracy"+str(args.k_spt)+" shot "+str(args.n_way)+" way.jpg")
    print("Test Accuracy === ",accs)
    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--eval',type = str, default = "",help = "Which dataset to evaluate")
    argparser.add_argument("--save_path",type = str,help = "Model saving path",default='/data4/home/manikantab/virtual_env/MAML-Pytorch/models')
    argparser.add_argument("--model_path",type = str,help = "Pretrained_model_path",default='')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    if args.eval != '':
        eval(args)
    else:
        main(args)
