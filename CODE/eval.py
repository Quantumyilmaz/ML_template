from utils.misc import make_dir,logprint,timer,start_logging
from configs.ML_config import cfg

def main():

    # DO STUFF
    df = ...        

    logprint("Exporting...")
    dirname = ...
    output_path = cfg.PATH.OUTPUT + f"/{dirname}"
    make_dir(output_path)
    filename = ...
    df.to_csv(output_path+f"/{filename}.txt", sep = " ", index = False)

if __name__ == "__main__":
    if cfg.LOG: start_logging()
    with timer("Evaluating..."):
        main()