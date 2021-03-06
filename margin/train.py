"""Train the model"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import margin.utils as utils
import margin.model.net as net
from margin.model import data_handler
import margin.model.data_loader as data_loader
from margin.evaluate import evaluate
from margin import logger

def create_output_dirs(outdir):
    """create ouput directory for log files and output images"""

    if os.path.isdir(outdir):
        logger.info("Output directory already exit from previous run")
    else:
        os.mkdir(outdir)
    for dirs in ["train", "test", "val"]:
        if not os.path.isdir(outdir+"/"+dirs):
            os.mkdir(outdir+"/"+dirs)

    outdir += "/"
    return outdir

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True,
                    help="directory with the data")
    p.add_argument("--params", type=str, dest="params", default="%s/DefaultParams.json" % os.path.dirname(__file__),
                    help="json file with model hyper parameters")
    p.add_argument('--model-dir', type=str, default="margin.out", 
            help="output directory, default is created in current working directory")
    p.add_argument('--restore-file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
    p.add_argument("--build", help="Build the dataset",
                   action="store_true", default=True)
    p.add_argument("--size", help="resize images to this size",
                   type=int, default=64)
    p.add_argument("--makedata", help="build datasets only",
                   action="store_true", default=False)
    
    p.add_argument("--skipbuild", help="skip split dataset",
                   action="store_true", default=False)

    return p

def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logger.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            model_dir, restore_file + '.pth.tar')
        logger.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logger.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logger.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

def main():
    """Main function."""
    
    parser = create_parser()
    args = parser.parse_args()
    GD = vars(args)

    logger.info('Input Options:')
    for key in GD.keys():
        logger.info('     %25s = %s' % (key, GD[key]))

    outdir = create_output_dirs(args.model_dir)

    # Load the hyper-parameters from json file
    json_path = GD['params']

    os.system("cp %s %s/params.json"%(json_path, outdir))

    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(outdir, 'train.log'))

    # Create the input data pipeline
    logger.info("Loading the datasets...")

    # fetch dataloaders
    
    if args.build and not args.skipbuild:
        data_handler.build_dataset(GD["data_dir"], outdir, size=GD['size'], seed=230)

    if args.makedata:
        sys.exit()

    dataloaders = data_loader.fetch_dataloader(params, outdir)
    
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logger.info("-............done.")

    # Define the model and optimizer
    model = net.ResNet18().cuda() if params.cuda else net.ResNet18()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, outdir,
                    args.restore_file)

    # Delete output
    # os.system("rm -rf %s/train"%outdir)
    # os.system("rm -rf %s/val" % outdir)
    # os.system("rm -rf %s/test"%outdir)

    os.system("mv margin.log %s/" % (outdir))
