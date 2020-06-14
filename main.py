import src.pipeline as pipe
from time import time

if __name__ == "__main__":
    start = time()
    keep_ordinal = pipe.Pipeline(model_to_use="XG")
    keep_ordinal.run()
    print(keep_ordinal.rmse)
    end = time()
    print(end - start)
