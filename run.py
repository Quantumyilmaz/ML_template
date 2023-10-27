import sys,os
project_path = os.path.abspath("./ML_template/CODE")
sys.path.append(project_path)

from utils.misc import logprint,start_logging,timer
from configs.misc_config import cfg as misc_cfg

from config import cfg

if __name__ == "__main__":
    try:
        if misc_cfg.LOG: start_logging()
        # TODO: find a good way to log all configurations
        # logprint("Executing program with the following configuration:")
        
        with timer("Executing program..."):

            if cfg.PREPROCESS:
                from preprocess import main as preprocess_main
                preprocess_main()
            if cfg.TUNE:
                from CODE.tune import main as tune_main
                tune_main()
            if cfg.TRAIN:
                from CODE.train import main as train_main
                train_main()

            from eval import main
            main()
            
    except Exception as e:
        logprint("Execution of the program failed due to the following error:")
        logprint(str(e))
