from configs.preprocess_config import cfg
from utils.prep import preprocess_data
from utils.misc import timer,start_logging
import pandas as pd
# from pqdm.processes import pqdm
from utils.parallelization import parallelize


def __get_data(filename,dtype=None,low_memory=True):
    return pd.read_csv(cfg.PATH.PREP + f"/{filename}.txt", sep = " ",dtype=dtype,low_memory=low_memory)


def __preprocess_data(item):
    with timer(f"Preprocessing..."):
        preprocess_data(df=__get_data(...,low_memory=False),item=item) # filename goes here. Can be specified in preprocess_config.py

def main():

    # Data Preprocessing
    iterable = [...] # in case the user needs to preprocess the data in different ways resulting in multiple preprocessings.
    if iterable is not None:
        if len(iterable) == 1:
            __preprocess_data(iterable[0])
        elif len(iterable) > 1:
            parallelize(__preprocess_data,iterable)

                
if __name__ == "__main__":
    if cfg.LOG: start_logging()
    with timer("Starting Preprocessing..."):
        main()