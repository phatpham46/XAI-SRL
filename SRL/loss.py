from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch

class CrossEntropyLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss'):
        super().__init__()
        
        self.alpha = alpha
        self.name = name
        self.ignore_index = -1
    def forward(self, inp, target, attnMasks = None):
        """
        This is the standard cross entropy loss as defined in pytorch.
        This loss should be used for single sentence or sentence pair classification tasks.

        To use this loss for training, set ``loss_type`` : **CrossEntropyLoss** in task file
        """
        loss = F.cross_entropy(inp, target, ignore_index=self.ignore_index) 
        loss *= self.alpha
        return loss

class NERLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss'):
        super().__init__()
        
        self.alpha = alpha
        self.name = name
        self.ignore_index = -1  #used to return 0 loss for such values
    def forward(self, inp, target, attnMasks = None):

        """
        This loss is a modified version of cross entropy loss for NER/sequence labelling tasks.
        This loss ignores extra ‘O’ values through attention masks.

        To use this loss for training, set ``loss_type`` : **NERLoss** in task file
        """
        
        '''
        inp shape would be (batchSize, maxSeqlen, classNum). But for loss calculation
        we need (batchSize, classNum). Hence we will squeeze the batchSize and maxSeqlen together.

        In NER, we have to ignore the loss created for the extra padding that
        has been done for making labels till max seq length. Hence we will use
        attention masks to ignore losses with those indices
        '''
        if attnMasks is not None:
            nerLoss = attnMasks.view(-1) == 1
            nerlogits = inp.view(-1, inp.size(-1))
            nerLabels = torch.where(
                nerLoss, target.view(-1), torch.tensor(self.ignore_index).type_as(target)
            )
            finalLoss = F.cross_entropy(nerlogits, nerLabels, ignore_index=self.ignore_index)

        else:
            finalLoss = F.cross_entropy(inp.view(-1, inp.size(-1)), target.view(-1),
                                        ignore_index=self.ignore_index)
 
        finalLoss *= self.alpha
        return finalLoss



