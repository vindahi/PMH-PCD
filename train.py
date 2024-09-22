import argparse
import time
import torch.nn.functional as F
from model import PCH, classone
from utils import *
from data import *


def train(args, dset):
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]
    logName = args.dataset + '_' + str(args.nbit)
    log = logger(logName)
    log.info('Training Stage...')
    log.info('mlpdrop: %f', (args.mlpdrop))
    log.info('drop: %f', (args.dropout))
    
    loss_l2 = torch.nn.MSELoss()

    model = PCH(args=args)
    model.train().cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}])

    start_time = time.time() * 1000

    DNPH = classone(args=args)
    DNPH.train().cuda()
    optimizer_loss = torch.optim.SGD(params=DNPH.parameters(), lr=1e-4)


    train_loader = data.DataLoader(my_dataset(dset.I_tr, dset.T_tr, dset.L_tr),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    

    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):
            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())

            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()
            label = label.cuda()
            nnbit = int(args.nbit)
           

            aff_label = torch.Tensor(aff_label).cuda()

            optimizer.zero_grad()
            optimizer_loss.zero_grad()
            fimage, ftext, H, pred = model(img_feat, txt_feat)
            H_norm = F.normalize(H)
            memory_image_centers = torch.zeros(args.classes, nnbit).cuda()
            memory_text_centers = torch.zeros(args.classes, nnbit).cuda()
            batch_image_centers = compute_centers(fimage, label.squeeze(), args.classes)
            batch_text_centers = compute_centers(ftext, label.squeeze(), args.classes)
            memory_image_centers = memory_image_centers.detach() + batch_image_centers
            memory_image_centers = F.normalize(memory_image_centers, dim=1)
            memory_text_centers = memory_text_centers.detach() + batch_text_centers
            memory_text_centers = F.normalize(memory_text_centers, dim=1)

            cluster_batch_loss = compute_cluster_loss(memory_image_centers, memory_text_centers, 0.007, label.squeeze(), args.classes)

            clf_loss = loss_l2(torch.sigmoid(pred), label)

            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)
            lossdn = DNPH(H, pred, label)
            lossqmi = qmi_loss(H, label)

            loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + lossdn * args.param_dn + lossqmi * args.param_qmi + cluster_batch_loss * args.param_cluster

            loss.backward()
            optimizer.step()
            optimizer_loss.step()
            if (i + 1) == len(train_loader) and (epoch + 1) % 10 == 0:
                log.info('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-S: %.4f, Loss-dn: %.4f, loss-qmi: %.4f, Loss-Clu: %.4f'
                          % (epoch + 1, args.epochs, loss.item(),
                             clf_loss.item() * args.param_clf,
                             similarity_loss.item() * args.param_sim,
                             lossdn.item() * args.param_dn,
                             lossqmi.item() * args.param_qmi,
                             cluster_batch_loss.item() * args.param_cluster))  

    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    log.info('Training Time: %.4f' % (elapsed))


    return model




def eval(model, dset, args):
    model.eval()
    logName = args.dataset + '_' + str(args.nbit)
    log = logger(logName)
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    retrieval_loader = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)

    retrievalP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(retrieval_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        _, _, H, _ = model(img_feat, txt_feat)
        retrievalP.append(H.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)

    end_time = time.time() * 1000
    retrieval_time = end_time - start_time

    log.info('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    valP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(val_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        _, _, H, _ = model(img_feat, txt_feat)
        valP.append(H.data.cpu().numpy())

    valH = np.concatenate(valP)
    valCode = np.sign(valH)


    end_time = time.time() * 1000
    query_time = end_time - start_time
    log.info('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))
    if args.save_flag:
        map = calculate_map(qu_B=valCode.astype(np.int8), re_B=retrievalCode.astype(np.int8), qu_L=dset.L_te, re_L=dset.L_db)
        log.info('[MAP] %.4f' % (map))
    return 0






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--model', type=str, default='FSFH', help='Use GMMH.')
    parser.add_argument('--epochs', type=int, default=140, help='Number of student epochs to train.')
    parser.add_argument('--epochs_pre', type=int, default=100, help='Epoch to learn the hashcode.')
    parser.add_argument('--nbit', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.8, help='')
    parser.add_argument('--mlpdrop', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)

    
    ## Data params
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1386)

    ## Net latent dimension params
    # COCO: 128 Flickr: 256
    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')
    

    ## Loss params
    parser.add_argument('--param_dn', type=float, default=0.000001)
    parser.add_argument('--param_qmi', type=float, default=0.000001)
    parser.add_argument('--param_clf', type=float, default=1)
    parser.add_argument('--param_sim', type=float, default=1)
    parser.add_argument('--param_cluster', type=float, default=0.01)

    ## Flag params
    parser.add_argument('--save_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    seed_setting(args.seed)

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))

    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]

    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)


    model = train(args, dset)
    eval(model, dset, args)