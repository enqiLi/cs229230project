import util as util
import torch
# from torchtext.data.metrics import bleu_score
import numpy as np
import nltk
import sys
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

class CaptioningSolver(object):
    """
    A CaptioningSolver encapsulates all the logic necessary for training
    image captioning models. The CaptioningSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolver works on a model object that must conform to the following
    API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(features, captions) must be a function that computes
      training-time loss and gradients, with the following inputs and outputs:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.device = kwargs.pop("device", 'cuda')
        print(self.device)
        self.gpu_ids = kwargs.pop("gpu_ids", [])
        self.save_dir = kwargs.pop("save_dir", "./save/")
        self.eval_steps = kwargs.pop("eval_steps", 500)
        self.optim = torch.optim.Adam(
            self.model.parameters(), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self.optim = torch.optim.Adam(
            self.model.parameters(), self.learning_rate)

        self._reset()

        self.idx_to_word = idx_to_word
        self.model = torch.nn.DataParallel(model, self.gpu_ids)
        self.model = self.model.to(self.device)

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.loss_history = []


    def _step(self, minibatch):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        # minibatch = util.create_minibatch(
        #    self.data, batch_size=self.batch_size, split="train"
        # )
        # captions = self.data['train_captions']
        # features = self.data['train_features']
        captions, features = minibatch
        captions = captions.to(self.device)
        features = features.to(self.device)
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        # print(captions.device)
        # print("in ", captions_in.device)
        mask = (captions_out != self.model.module._null).long()
        # print("mask has device, ", mask.device)
        t_features = features # torch.Tensor(features).to(self.device)
        t_captions_in = captions_in # torch.LongTensor(captions_in).to(self.device)

        t_captions_out = captions_out # torch.LongTensor(captions_out).to(self.device)
        t_mask = mask # torch.LongTensor(mask).to(self.device)
        logits = self.model(t_features, t_captions_in)

        loss = self.temporal_softmax_loss(
            logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Run optimization to train the model.
        """
        tbx = SummaryWriter(self.save_dir)
        steps_till_eval = self.eval_steps
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        step = 0
        epoch = 1
        batches_strings = torch.split(
            self.data["train_captions"], self.batch_size)
        batches_images = torch.split(
            self.data["train_features"], self.batch_size)
        batches = list(zip(batches_strings, batches_images))

        # for t in range(num_iterations):
        while epoch <= self.num_epochs:
            with torch.enable_grad(), \
                    tqdm(total=num_train) as progress_bar:
                for batch in batches:
                    self._step(batch)
                    step += self.batch_size
                    
                    progress_bar.update(self.batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             CELoss=self.loss_history[-1])
                    tbx.add_scalar('train/CELoss', self.loss_history[-1], step)
                    tbx.add_scalar('train/LR',
                               self.optim.param_groups[0]['lr'],
                               step)
                    
                    steps_till_eval -= self.batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = self.eval_steps

                        # Evaluate
                        tqdm.write(f'Evaluating at step {step}...')
                        dev_bleu = self.evaluate()
                        tbx.add_scalar('dev/bleu', dev_bleu, step)
            epoch += 1
    
    def evaluate(self):
        
        sample_codes = self.model.module.sample(self.data['dev_features'][:10].to(self.device), max_length=250)
        sample_codes = util.decode_codes(sample_codes, self.data['idx_to_word'])
        average_bleu = 0
        for i in range(len(sample_codes)):
            sample_caption = sample_codes[i]
            print(sample_caption)
            print('--------------')
            true_code = self.data['dev_codes'][i]
            print(true_code)
            print('--------------')
            average_bleu += nltk.translate.bleu_score.sentence_bleu([list(true_code)], list(sample_caption))
        average_bleu /= len(sample_codes)

        return average_bleu

    def temporal_softmax_loss(self, x, y, mask):
        """
        A temporal version of softmax loss for use in RNNs. We assume that we are
        making predictions over a vocabulary of size V for each timestep of a
        timeseries of length T, over a minibatch of size N. The input x gives scores
        for all vocabulary elements at all timesteps, and y gives the indices of the
        ground-truth element at each timestep. We use a cross-entropy loss at each
        timestep, summing the loss over all timesteps and averaging across the
        minibatch.

        As an additional complication, we may want to ignore the model output at some
        timesteps, since sequences of different length may have been combined into a
        minibatch and padded with NULL tokens. The optional mask argument tells us
        which elements should contribute to the loss.

        Inputs:
        - x: Input scores, of shape (N, T, V)
        - y: Ground-truth indices, of shape (N, T) where each element is in the range
             0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
          the scores at x[i, t] should contribute to the loss.

        Returns a tuple of:
        - loss: Scalar giving loss
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(
            x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss

