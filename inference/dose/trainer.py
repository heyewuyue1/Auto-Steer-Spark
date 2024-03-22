import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
from .database_util import collator
import os
import torch
from scipy.stats import pearsonr
from utils.custom_logging import logger
import memory_profiler
import time


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr


def eval_workload(workload, methods):

    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv('./data/imdb/workloads/{}.csv'.format(workload),sep='#',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)

    eval_score = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'],True)
    return eval_score, ds


def evaluate(model, ds, bs, norm, device, prints=True):
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))

            batch = batch.to(device)

            cost_preds = model(batch)
            cost_preds = cost_preds.squeeze()
            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())
    scores = print_qerror(norm.unnormalize_labels(cost_predss), ds.costs, prints)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    if prints:
        print('Corr: ',corr)
    return scores, corr

@memory_profiler.profile
def train(model, train_ds, val_ds, crit, \
    cost_norm, args, optimizer=None, scheduler=None):
    
    bs, device, epochs, clip_size = \
        args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)

    rng = np.random.default_rng()

    best_prev = 999999

    training_losses = []
    test_losses = []
    t0 = time.time()
    for epoch in range(epochs):
        losses = 0
        val_losses = 0
        cost_predss = np.empty(0)
        model.train()
        train_idxs = rng.permutation(len(train_ds))
        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            batch_cost_label = torch.FloatTensor(batch_labels).to(device)
            batch = batch.to(device)
            cost_preds = model(batch)
            cost_preds = cost_preds.squeeze()
            loss = crit(cost_preds, batch_cost_label)
            val_batch, val_batch_labels = collator(list(zip(*val_ds)))
            val_batch_cost_label = torch.FloatTensor(val_batch_labels).to(device)
            val_batch = val_batch.to(device)
            print('!!!')
            val_cost_preds = model(val_batch)
            val_cost_preds = val_cost_preds.squeeze()
            val_loss = crit(val_cost_preds, val_batch_cost_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()
            losses += loss.item()
            val_losses += val_loss.item()
            cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())
        test_scores, corrs = evaluate(model, val_ds, bs, cost_norm, device, False)
        if test_scores['q_mean'] < best_prev: ## mean mse
            best_prev = test_scores['q_mean']
        logger.info(f'Epoch {epoch} train. loss {losses / len(train_ds)}')
        logger.info(f'Epoch {epoch} val. loss {val_losses / len(val_ds)}')
        logger.info(f'Time: {time.time() - t0}')
        training_losses.append(losses / len(train_ds))
        test_losses.append(val_losses / len(val_ds))
        if len(training_losses) > 10 and training_losses[-1] < 0.1:
            last_two = np.min(training_losses[-2:])
            if last_two > training_losses[-10] or (training_losses[-10] - last_two < 0.0001):
                logger.info('Stopped training from convergence condition at epoch %s', epoch)
                break
        scheduler.step()   
    return training_losses, test_losses


def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']  