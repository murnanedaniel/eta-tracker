import torch
import numpy as np
import atexit
from ml_comm import get_rank, get_nranks, barrier, init, init_mpi, finalize, average, gradients, broadcast, check_buffers_match, config_team
from ml_comm import average as average_np

teamID=0
plugin_init=False
step = 0
import __main__ as main
# if hasattr(main, '__file__'):
#     init_mpi()


def average(s):
    average_np(np.array([s], dtype=np.float32))
    return s;


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, nstep, nteam, nthread_per_team, warmup, verb, freq):
        super(self.__class__, self).__init__(params)
        
        param = [pp.data.numpy() for p in self.param_groups
                 for pp in p['params']]
        self.maxmsglen = sum([p.size for p in param])
        self.nteam = nteam
        self.nthread_per_team = nthread_per_team
        
        global plugin_init
        if (plugin_init is False):
            init(self.nthread_per_team, self.nteam, self.maxmsglen)
            atexit.register(finalize)
            plugin_init=True
            

        global teamID;
        self.teamID = teamID
        teamID = teamID + 1

        config_team(self.teamID, 0, int(warmup*nstep), nstep, 2, 100)
        self.broadcast_parameters(0)


    def broadcast_parameters(self, root=0):
        """
        Broadcast of initial parameters
        """
      
        orig_param = [pp.data.numpy() for p in self.param_groups
                      for pp in p['params']]
        err = check_buffers_match(orig_param, 0)
        # If not consistent, we need to broadcast from rank 0 
        if err != 0:
            broadcast(orig_param, root)
            err = check_buffers_match(orig_param, 0)
            if (get_rank() == 0 and err == 0):
                print("Broadcast was successful")
        if (err != 0): 
            if (get_rank() == 0):
                print("Broadcast of initial parameters has not resulted in matching models")

    def average_gradients(self):
        """ Gradient averaging. """

        update_grads = [pp.grad.data.numpy() for p in self.param_groups
                        for pp in p['params'] if pp.requires_grad]
        new_grads = gradients(update_grads, self.teamID)
        for update,new in zip(update_grads, new_grads):
            update[:] = new

    def step(self, closure=None):
        self.average_gradients()
        return super(self.__class__, self).step(closure)



def DistributedOptimizer(optimizer, nstep, nteam=1, nthread_per_team=2, warmup=.10,
                         verb=2, freq=100):
    """
    An optimizer that wraps another torch.optim.Optimizer, using the Cray PE DL Plugin to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, nstep, nteam, nthread_per_team, warmup, verb, freq)
                                        
