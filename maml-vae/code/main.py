import pickle
import pprint
import json
import torch

# ----------------
import run
import arguments
import config.model_config

def load_mconf(config_path=None):

	print("loading model config ...")
	if config_path is not None:
		with open(config_path, 'r') as f:
			config_json = json.load(f)
			mconf = config.model_config.MAMLModelConfig()
			mconf.init_from_dict(config_json)
	else:
		# load default
		mconf = config.model_config.default_maml_mconf

	return mconf


def main():

	args = arguments.load_args()

	printer = pprint.PrettyPrinter(indent=4)

	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	if args.config_path != '':
		mconf = load_mconf(args.config_path)
	else:
		mconf = arguments.build_mconf_from_args(args)

	if args.online_inference:
		run.run_online_inference(
			mconf=mconf, timestamp=args.timestamp, 
			tgt_file=args.tgt_file, device=torch.device("cpu")
		)
	else:
		if not args.disable_gpu and torch.cuda.is_available():
			device = torch.device("cuda:{}".format(args.device_index))
		else:
			device = torch.device("cpu")
		print("[DEVICE INFO] using {}".format(device))

		run.run_maml(
			mconf=mconf, device=device, load_data=args.load_data,
			maml_epochs=args.maml_epochs, transfer_epochs=args.transfer_epochs,
			epochs_per_val=args.epochs_per_val, infer=args.inference,
			maml_batch_size=args.maml_batch_size, sub_batch_size=args.sub_batch_size, 
			train_batch_size=args.train_batch_size
		)
		with open("../config/{}.json".format(args.corpus), 'w') as f:
			json.dump(vars(mconf), f, indent=4)
			print("saved model config to ../config/{}.json".format(args.corpus))

	print(">>>>>>> Completed <<<<<<<")



if __name__ == "__main__":

	main()