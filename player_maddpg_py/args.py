class Argument(object):
	def __init__(self):
		# Environment
		# parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
		# parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
		# parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
		# parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
		self.good_policy = "maddpg"
		self.adv_policy = "maddpg"
		# Core training parameters
		self.lr =1e-2 # learning rate for Adam optimizer
		self.gamma = 0.95 # discount factor
		self.batch_size = 1024 # number of episodes to optimize at the same time
		self.num_units = 64 # number of units in the mlp
		# Checkpointing
		self.exp_name = None # name of the experiment
		self.save_dir = "./save_model/aiwc_maddpg" # directory in which training state and model should be saved
		self.save_rate = 1000 # save model once every time this many episodes are completed
		self.load_dir = "" # directory in which training state and model are loaded
		# Evaluation
		self.restore = False
		self.display = False
		self.benchmark = False
		self.benchmark_iters = 100000 # number of iterations run for benchmarking
		self.benchmark_dir = "./benchmark_files/" # directory where benchmark data is saved
		self.plots_dir = "./learning_curves/" # directory where plot data is saved