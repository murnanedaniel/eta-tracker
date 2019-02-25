from crayai import hpo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hpo_approach', type=str, default='genetic')
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--alloc_job_id', type=int)
parser.add_argument('--pop_size', type=int, default=16)
parser.add_argument('--generations', type=int, default=100)
parser.add_argument('--mutation_rate', type=float, default=0.05)
parser.add_argument('--crossover_rate', type=float, default=0.33)
args = parser.parse_args()

loss_funcs = ["binary_cross_entropy", "binary_cross_entropy_with_logits",
              "poisson_nll_loss", "cross_entropy",
              "kl_div", "l1_loss", "mse_loss"]
optimizers = ["Adam", "Adadelta", "Adagrad", "Adamax", "RMSprop", "SGD"]

params = hpo.Params([["--hidden_dim", 64, (4, 256)],
                     ["--n_iters", 4, (1, 8)],
                     ["--loss_func", "binary_cross_entropy", loss_funcs],
                     ["--optimizer", "Adam", optimizers],
                     ["--learning_rate", 0.001, (1e-6, 0.1)]])

cmd = "python ./train_dl.py configs/segclf_med.yaml --crayai-hpo"

evaluator = hpo.Evaluator(cmd, 
                          nodes=args.nodes, 
                          launcher='wlm', 
                          verbose=True, 
                          alloc_jobid=args.alloc_job_id)

if args.hpo_approach == "genetic":
  optimizer = hpo.genetic.Optimizer(evaluator,
                                    pop_size=args.pop_size,
                                    num_demes=1,
                                    generations=args.generations,
                                    mutation_rate=args.mutation_rate,
                                    crossover_rate=args.crossover_rate,
                                    verbose=True)
else:
  optimizer = hpo.random.Optimizer(evaluator, verbose=True)

optimizer.optimize(params)

